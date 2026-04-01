from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.types import ExperimentPaths


def _to_serializable(payload: Any) -> Any:
    """递归将 dataclass / Path 转成 JSON 可写格式。"""
    if is_dataclass(payload):
        return _to_serializable(asdict(payload))
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(k): _to_serializable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_to_serializable(v) for v in payload]
    return payload


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 方式写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def build_experiment_paths(
    outputs_root: str,
    problem_name: str,
    solver_name: str,
    run_name: str | None = None,
) -> ExperimentPaths:
    """按统一规范创建实验目录。"""
    root = Path(outputs_root)
    solver_root = root / problem_name / solver_name
    solver_root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{solver_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = solver_root / run_name
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"

    for path in (run_dir, tb_dir, ckpt_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    (solver_root / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")

    return ExperimentPaths(
        outputs_root=root,
        problem_name=problem_name,
        solver_name=solver_name,
        run_name=run_name,
        solver_root=solver_root,
        run_dir=run_dir,
        tb_dir=tb_dir,
        ckpt_dir=ckpt_dir,
        plots_dir=plots_dir,
        train_csv=run_dir / "train_metrics.csv",
        eval_csv=run_dir / "eval_metrics.csv",
        problem_config_json=run_dir / "config.problem.json",
        solver_config_json=run_dir / "config.solver.json",
        summary_json=run_dir / "summary.json",
        evaluation_summary_json=run_dir / "evaluation_summary.json",
    )


def resolve_latest_run(outputs_root: str, problem_name: str, solver_name: str) -> Path | None:
    """读取某个问题/求解器最近一次运行目录。"""
    latest_run_file = Path(outputs_root) / problem_name / solver_name / "latest_run.txt"
    if not latest_run_file.exists():
        return None
    return Path(latest_run_file.read_text(encoding="utf-8").strip())

