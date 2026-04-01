from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import problems  # noqa: F401
import solvers  # noqa: F401

from core.registry import get_problem_registration, get_solver_registration
from core.types import Metrics
from problems.ris_miso_urllc.rl_env import RISEnv
from solvers.rl.base import RLSolver


def override_dataclass(config: Any, overrides: dict[str, Any]):
    """只对 dataclass 中真实存在的字段应用覆盖。"""
    valid_fields = {item.name for item in fields(config)}
    payload = {key: value for key, value in overrides.items() if key in valid_fields and value is not None}
    if not payload:
        return config
    return replace(config, **payload)


def build_problem_bundle(problem_name: str, problem_config: Any | None = None) -> dict[str, Any]:
    """构造问题层组件集合。"""
    registration = get_problem_registration(problem_name)
    cfg = problem_config or registration.config_factory()
    return registration.builder(cfg)


def build_solver(problem_config: Any, solver_name: str, solver_config: Any | None = None):
    """构造求解器实例。"""
    registration = get_solver_registration(solver_name)
    cfg = solver_config or registration.config_factory()
    solver = registration.builder(problem_config, cfg)
    return solver, registration.category


def aggregate_metrics(metrics_list: list[Metrics]) -> dict[str, float]:
    """将多次评估结果汇总为平均指标。"""
    if not metrics_list:
        return {
            "avg_reward": 0.0,
            "avg_mean_sinr": 0.0,
            "avg_sum_cbl": 0.0,
            "avg_power": 0.0,
        }
    return {
        "avg_reward": float(sum(item.reward for item in metrics_list) / len(metrics_list)),
        "avg_mean_sinr": float(sum(item.mean_sinr for item in metrics_list) / len(metrics_list)),
        "avg_sum_cbl": float(sum(item.sum_cbl for item in metrics_list) / len(metrics_list)),
        "avg_power": float(sum(item.power for item in metrics_list) / len(metrics_list)),
    }


def evaluate_solver_over_instances(problem_name: str, problem_config: Any, solver, num_episodes: int) -> dict[str, Any]:
    """在多个问题实例上评估一个求解器。"""
    bundle = build_problem_bundle(problem_name, problem_config)
    sampler = bundle["sampler"]
    evaluator = bundle["evaluator"]

    metrics_list: list[Metrics] = []
    for _ in range(num_episodes):
        instance = sampler.sample()
        solution = solver.solve(instance)
        metrics_list.append(evaluator.evaluate(instance, solution))

    summary = aggregate_metrics(metrics_list)
    summary["episodes"] = int(num_episodes)
    return summary


def ensure_solver_bound_for_env(solver, env: RISEnv) -> None:
    """如果是 RL 求解器，则绑定状态维度与动作维度。"""
    if isinstance(solver, RLSolver) and solver.state_dim == 0:
        solver.bind_environment(env.state_dim, env.action_dim)


def infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    """从 checkpoint 路径反推实验目录。"""
    checkpoint_dir = checkpoint_path.parent
    return checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """将 dataclass 配置转成字典。"""
    return asdict(obj)
