from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from configs.default import DEFAULT_PROBLEM_NAME, DEFAULT_SOLVER_NAME, build_problem_config, build_solver_config, list_registered_solvers
from core.io import build_experiment_paths, write_json
from core.logging import ExperimentLogger
from core.seeds import set_global_seed
from problems.ris_miso_urllc.rl_env import RISEnv
from runners.common import build_problem_bundle, build_solver, dataclass_to_dict, ensure_solver_bound_for_env, evaluate_solver_over_instances, override_dataclass
from solvers.rl.base import RLSolver
from solvers.rl.replay_buffer import ReplayBuffer


def parse_args(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> argparse.Namespace:
    """解析统一训练入口参数。"""
    parser = argparse.ArgumentParser(description="统一求解器训练/运行入口")
    parser.add_argument("--problem", type=str, default=default_problem, help="问题名称")
    parser.add_argument("--solver", type=str, default=default_solver, choices=list_registered_solvers(), help="求解器名称")
    parser.add_argument("--run-name", type=str, default=None, help="实验运行名称")
    parser.add_argument("--outputs-root", type=str, default=None, help="覆盖输出根目录")

    parser.add_argument("--seed", type=int, default=None, help="覆盖问题随机种子")
    parser.add_argument("--train-episodes", type=int, default=None, help="覆盖训练回合数")
    parser.add_argument("--eval-episodes", type=int, default=None, help="覆盖评估回合数")
    parser.add_argument("--eval-interval", type=int, default=None, help="覆盖评估间隔")
    parser.add_argument("--save-interval", type=int, default=None, help="覆盖保存间隔")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖单场景最大优化步数")

    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--buffer-size", type=int, default=None, help="覆盖经验回放容量")
    parser.add_argument("--warmup-episodes", type=int, default=None, help="覆盖 warmup 回合数")
    parser.add_argument("--actor-lr", type=float, default=None, help="覆盖 actor 学习率")
    parser.add_argument("--critic-lr", type=float, default=None, help="覆盖 critic 学习率")
    parser.add_argument("--device", type=str, default=None, help="覆盖训练设备")
    parser.add_argument("--num-candidates", type=int, default=None, help="随机搜索候选解数量")
    return parser.parse_args()


def build_configs(args: argparse.Namespace):
    """根据命令行覆盖项构造问题配置与求解器配置。"""
    problem_config = build_problem_config(args.problem)
    solver_config = build_solver_config(args.solver)

    problem_config = override_dataclass(
        problem_config,
        {
            "seed": args.seed,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
            "max_steps": args.max_steps,
            "outputs_root": args.outputs_root,
        },
    )
    solver_config = override_dataclass(
        solver_config,
        {
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "warmup_episodes": args.warmup_episodes,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "device": args.device,
            "num_candidates": args.num_candidates,
        },
    )
    return problem_config, solver_config


def train_rl_solver(problem_name: str, problem_config, solver, paths) -> dict[str, object]:
    """统一 RL 训练主循环。"""
    bundle = build_problem_bundle(problem_name, problem_config)
    env = bundle["env_cls"](problem_config, sampler=bundle["sampler"], evaluator=bundle["evaluator"], action_codec=bundle["action_codec"], observation_encoder=bundle["observation_encoder"])
    initial_state = env.reset()
    ensure_solver_bound_for_env(solver, env)

    replay_buffer = ReplayBuffer(state_dim=env.state_dim, action_dim=env.action_dim, capacity=solver.solver_config.buffer_size)
    rng = np.random.default_rng(problem_config.seed + 7)

    logger = ExperimentLogger(
        paths=paths,
        train_fieldnames=["episode", "reward", "mean_sinr", "sum_cbl", "total_power", "critic_loss", "actor_loss", "buffer_size"],
        eval_fieldnames=["episode", "avg_reward", "avg_mean_sinr", "avg_sum_cbl", "avg_power"],
    )

    best_eval_reward = float("-inf")
    last_eval_metrics: dict[str, float] | None = None
    last_eval_episode = 0

    print(f"开始训练: problem={problem_name}, solver={solver.name}, state_dim={env.state_dim}, action_dim={env.action_dim}, device={solver.device}, run_dir={paths.run_dir}")

    try:
        for episode in range(1, problem_config.train_episodes + 1):
            state = env.reset()
            episode_reward = 0.0
            running_critic_loss = float("nan")
            running_actor_loss = float("nan")
            last_info = {
                "sinr": np.zeros(problem_config.K),
                "cbl": np.zeros(problem_config.K),
                "power": 0.0,
            }

            for _ in range(problem_config.max_steps):
                if episode <= getattr(solver.solver_config, "warmup_episodes", 0):
                    action = bundle["action_codec"].sample_random_action(rng)
                else:
                    action = solver.select_action(state)
                    noise = rng.normal(0.0, solver.solver_config.exploration_noise, size=env.action_dim).astype(np.float32)
                    action = np.clip(action + noise, -1.0, 1.0)

                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward * solver.solver_config.reward_scale, next_state, done)

                state = next_state
                episode_reward += float(reward)
                last_info = info

                losses = solver.update(replay_buffer)
                if losses is not None:
                    running_critic_loss = losses["critic_loss"]
                    if losses["actor_updated"] > 0.5:
                        running_actor_loss = losses["actor_loss"]

                if done:
                    break

            mean_sinr = float(np.mean(last_info["sinr"]))
            sum_cbl = float(np.sum(last_info["cbl"]))
            total_power = float(last_info["power"])

            train_scalars = {
                "episode_reward": episode_reward,
                "mean_sinr": mean_sinr,
                "sum_cbl": sum_cbl,
                "total_power": total_power,
            }
            logger.add_scalars("train", train_scalars, episode)
            if not np.isnan(running_critic_loss):
                logger.add_scalars("train", {"critic_loss": float(running_critic_loss)}, episode)
            if not np.isnan(running_actor_loss):
                logger.add_scalars("train", {"actor_loss": float(running_actor_loss)}, episode)

            logger.write_train_row(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "mean_sinr": mean_sinr,
                    "sum_cbl": sum_cbl,
                    "total_power": total_power,
                    "critic_loss": "" if np.isnan(running_critic_loss) else float(running_critic_loss),
                    "actor_loss": "" if np.isnan(running_actor_loss) else float(running_actor_loss),
                    "buffer_size": len(replay_buffer),
                }
            )

            if episode % max(1, problem_config.eval_interval) == 0:
                metrics = evaluate_solver_over_instances(problem_name, problem_config, solver, problem_config.eval_episodes)
                last_eval_metrics = metrics
                last_eval_episode = episode
                logger.write_eval_row({"episode": episode, "avg_reward": metrics["avg_reward"], "avg_mean_sinr": metrics["avg_mean_sinr"], "avg_sum_cbl": metrics["avg_sum_cbl"], "avg_power": metrics["avg_power"]})
                logger.add_scalars("eval", {k: float(v) for k, v in metrics.items() if k.startswith("avg_")}, episode)
                if metrics["avg_reward"] > best_eval_reward:
                    best_eval_reward = metrics["avg_reward"]
                    solver.save(paths.ckpt_dir / "best.pt")
                print(f"[Eval] episode={episode} avg_reward={metrics['avg_reward']:.4f} avg_mean_sinr={metrics['avg_mean_sinr']:.6e} avg_sum_cbl={metrics['avg_sum_cbl']:.4f}")

            if episode % max(1, problem_config.save_interval) == 0:
                solver.save(paths.ckpt_dir / "latest.pt")

            if episode == 1 or episode % 100 == 0:
                print(f"[Train] episode={episode} reward={episode_reward:.4f} mean_sinr={mean_sinr:.6e} critic_loss={running_critic_loss:.4f} actor_loss={running_actor_loss:.4f} buffer={len(replay_buffer)}")

        if last_eval_episode != problem_config.train_episodes:
            last_eval_metrics = evaluate_solver_over_instances(problem_name, problem_config, solver, problem_config.eval_episodes)
            last_eval_episode = problem_config.train_episodes
            logger.write_eval_row({"episode": problem_config.train_episodes, "avg_reward": last_eval_metrics["avg_reward"], "avg_mean_sinr": last_eval_metrics["avg_mean_sinr"], "avg_sum_cbl": last_eval_metrics["avg_sum_cbl"], "avg_power": last_eval_metrics["avg_power"]})
            logger.add_scalars("eval", {k: float(v) for k, v in last_eval_metrics.items() if k.startswith("avg_")}, problem_config.train_episodes)
            if last_eval_metrics["avg_reward"] > best_eval_reward:
                best_eval_reward = last_eval_metrics["avg_reward"]
                solver.save(paths.ckpt_dir / "best.pt")
    finally:
        solver.save(paths.ckpt_dir / "latest.pt")
        if not (paths.ckpt_dir / "best.pt").exists():
            solver.save(paths.ckpt_dir / "best.pt")

    summary = {
        "problem_name": problem_name,
        "solver_name": solver.name,
        "run_dir": str(paths.run_dir.resolve()),
        "resolved_device": str(solver.device),
        "train_episodes": problem_config.train_episodes,
        "best_eval_reward": None if best_eval_reward == float("-inf") else float(best_eval_reward),
        "last_eval_episode": last_eval_episode,
        "last_eval_metrics": last_eval_metrics,
        "train_metrics_csv": str(paths.train_csv.resolve()),
        "eval_metrics_csv": str(paths.eval_csv.resolve()),
        "tensorboard_dir": str(paths.tb_dir.resolve()),
        "latest_checkpoint": str((paths.ckpt_dir / "latest.pt").resolve()),
        "best_checkpoint": str((paths.ckpt_dir / "best.pt").resolve()),
    }
    logger.write_summary(summary)
    logger.close()
    return summary


def run_deterministic_solver(problem_name: str, problem_config, solver, paths) -> dict[str, object]:
    """对非 RL 求解器执行统一评估式运行。"""
    logger = ExperimentLogger(
        paths=paths,
        train_fieldnames=["episode", "reward", "mean_sinr", "sum_cbl", "total_power", "critic_loss", "actor_loss", "buffer_size"],
        eval_fieldnames=["episode", "avg_reward", "avg_mean_sinr", "avg_sum_cbl", "avg_power"],
    )

    bundle = build_problem_bundle(problem_name, problem_config)
    sampler = bundle["sampler"]
    evaluator = bundle["evaluator"]
    best_reward = float("-inf")

    print(f"开始运行: problem={problem_name}, solver={solver.name}, run_dir={paths.run_dir}")

    for episode in range(1, problem_config.train_episodes + 1):
        instance = sampler.sample()
        solution = solver.solve(instance)
        metrics = evaluator.evaluate(instance, solution)
        log_row = metrics.to_log_dict()
        logger.write_train_row(
            {
                "episode": episode,
                "reward": log_row["reward"],
                "mean_sinr": log_row["mean_sinr"],
                "sum_cbl": log_row["sum_cbl"],
                "total_power": log_row["total_power"],
                "critic_loss": "",
                "actor_loss": "",
                "buffer_size": 0,
            }
        )
        logger.add_scalars("train", log_row, episode)
        if metrics.reward > best_reward:
            best_reward = metrics.reward
            solver.save(paths.ckpt_dir / "best.pt")
        if episode % max(1, problem_config.eval_interval) == 0:
            eval_metrics = evaluate_solver_over_instances(problem_name, problem_config, solver, problem_config.eval_episodes)
            logger.write_eval_row({"episode": episode, "avg_reward": eval_metrics["avg_reward"], "avg_mean_sinr": eval_metrics["avg_mean_sinr"], "avg_sum_cbl": eval_metrics["avg_sum_cbl"], "avg_power": eval_metrics["avg_power"]})
            logger.add_scalars("eval", {k: float(v) for k, v in eval_metrics.items() if k.startswith("avg_")}, episode)

    solver.save(paths.ckpt_dir / "latest.pt")
    if not (paths.ckpt_dir / "best.pt").exists():
        solver.save(paths.ckpt_dir / "best.pt")

    summary = {
        "problem_name": problem_name,
        "solver_name": solver.name,
        "run_dir": str(paths.run_dir.resolve()),
        "best_eval_reward": float(best_reward),
        "train_metrics_csv": str(paths.train_csv.resolve()),
        "eval_metrics_csv": str(paths.eval_csv.resolve()),
        "latest_checkpoint": str((paths.ckpt_dir / "latest.pt").resolve()),
        "best_checkpoint": str((paths.ckpt_dir / "best.pt").resolve()),
    }
    logger.write_summary(summary)
    logger.close()
    return summary


def main(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> None:
    args = parse_args(default_problem=default_problem, default_solver=default_solver)
    problem_config, solver_config = build_configs(args)
    set_global_seed(problem_config.seed)

    paths = build_experiment_paths(problem_config.outputs_root, args.problem, args.solver, args.run_name)
    write_json(paths.problem_config_json, dataclass_to_dict(problem_config))
    write_json(paths.solver_config_json, dataclass_to_dict(solver_config))

    solver, category = build_solver(problem_config, args.solver, solver_config)
    if category == "rl":
        train_rl_solver(args.problem, problem_config, solver, paths)
    else:
        run_deterministic_solver(args.problem, problem_config, solver, paths)


if __name__ == "__main__":
    main()

