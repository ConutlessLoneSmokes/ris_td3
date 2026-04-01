from __future__ import annotations

import argparse
from pathlib import Path

from configs.default import DEFAULT_PROBLEM_NAME, DEFAULT_SOLVER_NAME, build_problem_config, build_solver_config, list_registered_solvers
from core.io import resolve_latest_run, write_json
from runners.common import build_problem_bundle, build_solver, dataclass_to_dict, ensure_solver_bound_for_env, evaluate_solver_over_instances, infer_run_dir_from_checkpoint, override_dataclass
from solvers.rl.base import RLSolver


def parse_args(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> argparse.Namespace:
    """解析统一评估入口参数。"""
    parser = argparse.ArgumentParser(description="统一求解器评估入口")
    parser.add_argument("--problem", type=str, default=default_problem, help="问题名称")
    parser.add_argument("--solver", type=str, default=default_solver, choices=list_registered_solvers(), help="求解器名称")
    parser.add_argument("--checkpoint", type=str, default=None, help="RL 求解器 checkpoint 路径")
    parser.add_argument("--run-dir", type=str, default=None, help="显式指定实验目录")
    parser.add_argument("--eval-episodes", type=int, default=None, help="覆盖评估回合数")
    parser.add_argument("--outputs-root", type=str, default=None, help="覆盖输出根目录")
    parser.add_argument("--num-candidates", type=int, default=None, help="随机搜索候选解数量")
    return parser.parse_args()


def main(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> None:
    args = parse_args(default_problem=default_problem, default_solver=default_solver)

    problem_config = build_problem_config(args.problem)
    solver_config = build_solver_config(args.solver)
    problem_config = override_dataclass(problem_config, {"eval_episodes": args.eval_episodes, "outputs_root": args.outputs_root})
    solver_config = override_dataclass(solver_config, {"num_candidates": args.num_candidates})

    solver, category = build_solver(problem_config, args.solver, solver_config)

    checkpoint_path: Path | None = None
    run_dir: Path | None = Path(args.run_dir) if args.run_dir else None
    if category == "rl":
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
        elif run_dir is not None:
            checkpoint_path = run_dir / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        else:
            latest_run = resolve_latest_run(problem_config.outputs_root, args.problem, args.solver)
            if latest_run is None:
                raise FileNotFoundError(f"未找到 {args.problem}/{args.solver} 的 latest_run.txt")
            run_dir = latest_run
            checkpoint_path = run_dir / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = run_dir / "checkpoints" / "latest.pt"

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

        bundle = build_problem_bundle(args.problem, problem_config)
        env = bundle["env_cls"](problem_config, sampler=bundle["sampler"], evaluator=bundle["evaluator"], action_codec=bundle["action_codec"], observation_encoder=bundle["observation_encoder"])
        ensure_solver_bound_for_env(solver, env)
        solver.load(checkpoint_path)
        if run_dir is None:
            run_dir = infer_run_dir_from_checkpoint(checkpoint_path)

    summary = evaluate_solver_over_instances(args.problem, problem_config, solver, problem_config.eval_episodes)
    summary.update(
        {
            "problem_name": args.problem,
            "solver_name": args.solver,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
            "solver_config": dataclass_to_dict(solver_config),
        }
    )

    print(f"problem: {args.problem}")
    print(f"solver: {args.solver}")
    if checkpoint_path is not None:
        print(f"checkpoint: {checkpoint_path}")
    print(f"episodes: {problem_config.eval_episodes}")
    print(f"avg_reward: {summary['avg_reward']:.6f}")
    print(f"avg_mean_sinr: {summary['avg_mean_sinr']:.6e}")
    print(f"avg_sum_cbl: {summary['avg_sum_cbl']:.6f}")
    print(f"avg_power: {summary['avg_power']:.6f}")

    if run_dir is not None:
        write_json(run_dir / "evaluation_summary.json", summary)


if __name__ == "__main__":
    main()
