from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.replay_buffer import ReplayBuffer
from agents.td3 import TD3Agent
from configs.default import SystemConfig
from envs.ris_env import RISEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIS-TD3 训练脚本")
    parser.add_argument("--run-name", type=str, default=None, help="实验运行名称，默认自动生成")
    parser.add_argument("--train-episodes", type=int, default=None, help="覆盖默认训练回合数")
    parser.add_argument("--warmup-episodes", type=int, default=None, help="覆盖默认随机探索回合数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖默认 batch size")
    parser.add_argument("--buffer-size", type=int, default=None, help="覆盖默认回放缓存容量")
    parser.add_argument("--eval-interval", type=int, default=None, help="覆盖默认评估间隔")
    parser.add_argument("--save-interval", type=int, default=None, help="覆盖默认保存间隔")
    parser.add_argument("--eval-episodes", type=int, default=None, help="覆盖默认评估回合数")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SystemConfig:
    cfg = SystemConfig()
    overrides = {
        "train_episodes": args.train_episodes,
        "warmup_episodes": args.warmup_episodes,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "eval_episodes": args.eval_episodes,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg = replace(cfg, **{key: value})
    return cfg


def build_run_paths(cfg: SystemConfig, run_name: str | None) -> dict[str, Path]:
    outputs_root = Path(cfg.outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"td3_ris_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = outputs_root / run_name
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"

    for path in (run_dir, tb_dir, ckpt_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    (outputs_root / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    return {
        "outputs_root": outputs_root,
        "run_dir": run_dir,
        "tb_dir": tb_dir,
        "ckpt_dir": ckpt_dir,
        "plots_dir": plots_dir,
        "train_csv": run_dir / "train_metrics.csv",
        "eval_csv": run_dir / "eval_metrics.csv",
        "config_json": run_dir / "config.json",
        "summary_json": run_dir / "summary.json",
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_evaluation(agent: TD3Agent, cfg: SystemConfig, num_episodes: int) -> dict[str, float]:
    env = RISEnv(cfg)

    rewards = []
    mean_sinrs = []
    sum_cbls = []
    total_powers = []

    for _ in range(num_episodes):
        state = env.reset()
        action = agent.select_action(state, deterministic=True)
        _, reward, _, info = env.step(action)

        rewards.append(float(reward))
        mean_sinrs.append(float(np.mean(info["sinr"])))
        sum_cbls.append(float(np.sum(info["cbl"])))
        total_powers.append(float(info["power"]))

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_mean_sinr": float(np.mean(mean_sinrs)),
        "avg_sum_cbl": float(np.mean(sum_cbls)),
        "avg_power": float(np.mean(total_powers)),
    }


def evaluate_policy(agent: TD3Agent, cfg: SystemConfig, num_episodes: int) -> dict[str, float]:
    return run_evaluation(agent, cfg, num_episodes)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    paths = build_run_paths(cfg, args.run_name)
    cfg = replace(cfg, log_dir=str(paths["tb_dir"]), ckpt_dir=str(paths["ckpt_dir"]))

    config_payload = asdict(cfg)
    config_payload["paper_alignment_notes"] = {
        "actor_hidden_dims": "论文给出 [800, 400, 200]",
        "critic_architecture": "状态分支 800、动作分支 800，融合后 [600, 400]，并使用 LayerNorm",
        "learning_rate": "论文给出 actor/critic 均为 1e-4",
        "buffer_batch": "论文给出 replay buffer=10000, batch=64",
        "noise_variance": "论文给出探索噪声和 target noise 的方差均为 0.1，因此代码中使用标准差 sqrt(0.1)",
        "assumptions": {
            "gamma": "论文未报告具体数值，这里保留 0.99；当前 single-shot 环境下 done=True，训练中不会实际使用未来回报",
            "noise_clip": "论文给出存在截断常数 c，但未报告具体数值，这里保留 TD3 常用值 0.5",
        },
    }
    write_json(paths["config_json"], config_payload)

    writer = SummaryWriter(log_dir=str(paths["tb_dir"]))
    env = RISEnv(cfg)
    initial_state = env.reset()
    agent = TD3Agent(state_dim=initial_state.shape[0], action_dim=env.action_dim, cfg=cfg)
    replay_buffer = ReplayBuffer(state_dim=env.state_dim, action_dim=env.action_dim, capacity=cfg.buffer_size)
    rng = np.random.default_rng(cfg.seed + 7)

    best_eval_reward = -np.inf
    last_eval_metrics: dict[str, float] | None = None
    last_eval_episode = 0

    train_csv_file = paths["train_csv"].open("w", encoding="utf-8", newline="")
    eval_csv_file = paths["eval_csv"].open("w", encoding="utf-8", newline="")
    train_writer = csv.DictWriter(
        train_csv_file,
        fieldnames=[
            "episode",
            "reward",
            "mean_sinr",
            "sum_cbl",
            "total_power",
            "critic_loss",
            "actor_loss",
            "buffer_size",
        ],
    )
    eval_writer = csv.DictWriter(
        eval_csv_file,
        fieldnames=["episode", "avg_reward", "avg_mean_sinr", "avg_sum_cbl", "avg_power"],
    )
    train_writer.writeheader()
    eval_writer.writeheader()

    print(
        f"开始训练: state_dim={env.state_dim}, action_dim={env.action_dim}, "
        f"device={agent.device}, run_dir={paths['run_dir']}"
    )

    try:
        for episode in range(1, cfg.train_episodes + 1):
            state = env.reset()
            episode_reward = 0.0
            
            last_info = {
                "sinr": np.zeros(cfg.K),
                "cbl": np.zeros(cfg.K),
                "theta": np.zeros(cfg.N),
                "power": 0.0
            }
            last_losses = None
            
            for t in range(cfg.max_steps):
                if agent.total_it <= cfg.warmup_episodes:
                    action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                else:
                    action = agent.select_action(state)
                    noise = rng.normal(0.0, cfg.exploration_noise, size=env.action_dim).astype(np.float32)
                    action = np.clip(action + noise, -1.0, 1.0)

                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                last_info = info

                losses = agent.train_step(replay_buffer)
                if losses is not None:
                    last_losses = losses
                
                if done:
                    break

            mean_sinr = float(np.mean(last_info["sinr"]))
            sum_cbl = float(np.sum(last_info["cbl"]))
            total_power = float(last_info["power"])

            writer.add_scalar("train/episode_reward", episode_reward, episode)
            writer.add_scalar("train/mean_sinr", mean_sinr, episode)
            writer.add_scalar("train/sum_cbl", sum_cbl, episode)
            writer.add_scalar("train/total_power", total_power, episode)

            critic_loss_val = ""
            actor_loss_val = ""

            if last_losses is not None:
                critic_loss_val = last_losses["critic_loss"]
                writer.add_scalar("train/critic_loss", critic_loss_val, episode)
                if last_losses["actor_updated"] > 0.5:
                    actor_loss_val = last_losses["actor_loss"]
                    writer.add_scalar("train/actor_loss", actor_loss_val, episode)

            train_writer.writerow(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "mean_sinr": mean_sinr,
                    "sum_cbl": sum_cbl,
                    "total_power": total_power,
                    "critic_loss": critic_loss_val,
                    "actor_loss": actor_loss_val,
                    "buffer_size": len(replay_buffer),
                }
            )

            if episode % max(1, cfg.eval_interval) == 0:
                metrics = evaluate_policy(agent, cfg, cfg.eval_episodes)
                last_eval_metrics = metrics
                last_eval_episode = episode
                writer.add_scalar("eval/avg_reward", metrics["avg_reward"], episode)
                writer.add_scalar("eval/avg_mean_sinr", metrics["avg_mean_sinr"], episode)
                writer.add_scalar("eval/avg_sum_cbl", metrics["avg_sum_cbl"], episode)
                writer.add_scalar("eval/avg_power", metrics["avg_power"], episode)
                eval_writer.writerow({"episode": episode, **metrics})

                if metrics["avg_reward"] > best_eval_reward:
                    best_eval_reward = metrics["avg_reward"]
                    agent.save(paths["ckpt_dir"] / "best.pt")

                print(
                    f"[Eval] episode={episode} "
                    f"avg_reward={metrics['avg_reward']:.4f} "
                    f"avg_mean_sinr={metrics['avg_mean_sinr']:.4f} "
                    f"avg_sum_cbl={metrics['avg_sum_cbl']:.4f}"
                )

            if episode % max(1, cfg.save_interval) == 0:
                agent.save(paths["ckpt_dir"] / "latest.pt")

            if episode == 1 or episode % 100 == 0:
                c_loss = float("nan") if critic_loss_val == "" else critic_loss_val
                a_loss = float("nan") if actor_loss_val == "" else actor_loss_val
                print(
                    f"[Train] episode={episode} reward={episode_reward:.4f} "
                    f"mean_sinr={mean_sinr:.4f} critic_loss={c_loss:.4f} "
                    f"actor_loss={a_loss:.4f} buffer={len(replay_buffer)}"
                )

        if last_eval_episode != cfg.train_episodes:
            last_eval_metrics = evaluate_policy(agent, cfg, cfg.eval_episodes)
            last_eval_episode = cfg.train_episodes
            writer.add_scalar("eval/avg_reward", last_eval_metrics["avg_reward"], cfg.train_episodes)
            writer.add_scalar("eval/avg_mean_sinr", last_eval_metrics["avg_mean_sinr"], cfg.train_episodes)
            writer.add_scalar("eval/avg_sum_cbl", last_eval_metrics["avg_sum_cbl"], cfg.train_episodes)
            writer.add_scalar("eval/avg_power", last_eval_metrics["avg_power"], cfg.train_episodes)
            eval_writer.writerow({"episode": cfg.train_episodes, **last_eval_metrics})
            if last_eval_metrics["avg_reward"] > best_eval_reward:
                best_eval_reward = last_eval_metrics["avg_reward"]
                agent.save(paths["ckpt_dir"] / "best.pt")
    finally:
        agent.save(paths["ckpt_dir"] / "latest.pt")
        if not (paths["ckpt_dir"] / "best.pt").exists():
            agent.save(paths["ckpt_dir"] / "best.pt")
        summary = {
            "run_dir": str(paths["run_dir"].resolve()),
            "resolved_device": str(agent.device),
            "train_episodes": cfg.train_episodes,
            "warmup_episodes": cfg.warmup_episodes,
            "best_eval_reward": None if best_eval_reward == -np.inf else float(best_eval_reward),
            "last_eval_episode": last_eval_episode,
            "last_eval_metrics": last_eval_metrics,
            "train_metrics_csv": str(paths["train_csv"].resolve()),
            "eval_metrics_csv": str(paths["eval_csv"].resolve()),
            "tensorboard_dir": str(paths["tb_dir"].resolve()),
            "latest_checkpoint": str((paths["ckpt_dir"] / "latest.pt").resolve()),
            "best_checkpoint": str((paths["ckpt_dir"] / "best.pt").resolve()),
        }
        write_json(paths["summary_json"], summary)
        train_csv_file.close()
        eval_csv_file.close()
        writer.close()

if __name__ == "__main__":
    main()
