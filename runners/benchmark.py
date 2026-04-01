from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from configs.default import DEFAULT_PROBLEM_NAME, build_problem_config, build_solver_config, list_registered_solvers
from core.io import resolve_latest_run, write_json
from runners.common import build_solver, build_problem_bundle, dataclass_to_dict, ensure_solver_bound_for_env, evaluate_solver_over_instances


def parse_args() -> argparse.Namespace:
    """解析 benchmark 入口参数。"""
    parser = argparse.ArgumentParser(description="多求解器 benchmark 入口")
    parser.add_argument("--problem", type=str, default=DEFAULT_PROBLEM_NAME, help="问题名称")
    parser.add_argument("--solvers", type=str, default="td3,ddpg,random_search", help="逗号分隔的求解器列表")
    parser.add_argument("--eval-episodes", type=int, default=None, help="评估回合数")
    parser.add_argument("--outputs-root", type=str, default=None, help="覆盖输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_config = build_problem_config(args.problem)
    if args.eval_episodes is not None:
        problem_config.eval_episodes = args.eval_episodes
    if args.outputs_root is not None:
        problem_config.outputs_root = args.outputs_root

    results: list[dict[str, object]] = []
    for solver_name in [item.strip() for item in args.solvers.split(",") if item.strip()]:
        solver_config = build_solver_config(solver_name)
        solver, category = build_solver(problem_config, solver_name, solver_config)
        checkpoint_path = None

        if category == "rl":
            latest_run = resolve_latest_run(problem_config.outputs_root, args.problem, solver_name)
            if latest_run is None:
                print(f"[Skip] 未找到 {solver_name} 的最新运行目录，跳过。")
                continue
            checkpoint_path = latest_run / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = latest_run / "checkpoints" / "latest.pt"
            if not checkpoint_path.exists():
                print(f"[Skip] 未找到 {solver_name} 的 checkpoint，跳过。")
                continue

            bundle = build_problem_bundle(args.problem, problem_config)
            env = bundle["env_cls"](problem_config, sampler=bundle["sampler"], evaluator=bundle["evaluator"], action_codec=bundle["action_codec"], observation_encoder=bundle["observation_encoder"])
            ensure_solver_bound_for_env(solver, env)
            solver.load(checkpoint_path)

        summary = evaluate_solver_over_instances(args.problem, problem_config, solver, problem_config.eval_episodes)
        summary.update(
            {
                "problem_name": args.problem,
                "solver_name": solver_name,
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
                "solver_config": dataclass_to_dict(solver_config),
            }
        )
        results.append(summary)
        print(f"[Benchmark] solver={solver_name} avg_reward={summary['avg_reward']:.6f} avg_mean_sinr={summary['avg_mean_sinr']:.6e}")

    benchmark_root = Path(problem_config.outputs_root) / args.problem / "benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    output_path = benchmark_root / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(output_path, {"problem_name": args.problem, "results": results})
    print(f"benchmark_summary: {output_path}")


if __name__ == "__main__":
    main()
