from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Metrics:
    """统一保存一次求解/交互后的评估结果。"""

    reward: float
    sinr: np.ndarray
    cbl: np.ndarray
    power: float
    constraint_violations: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def mean_sinr(self) -> float:
        """返回所有用户 SINR 的均值。"""
        return float(np.mean(self.sinr))

    @property
    def sum_cbl(self) -> float:
        """返回总信道块长。"""
        return float(np.sum(self.cbl))

    def to_log_dict(self) -> dict[str, float]:
        """转换为便于 CSV / TensorBoard 记录的扁平字典。"""
        payload = {
            "reward": float(self.reward),
            "mean_sinr": self.mean_sinr,
            "sum_cbl": self.sum_cbl,
            "total_power": float(self.power),
        }
        payload.update({f"violation_{k}": float(v) for k, v in self.constraint_violations.items()})
        payload.update({str(k): float(v) for k, v in self.extra.items() if np.isscalar(v)})
        return payload

    def to_serializable(self) -> dict[str, Any]:
        """转换为可直接写入 JSON 的结构。"""
        return {
            "reward": float(self.reward),
            "sinr": np.asarray(self.sinr, dtype=float).tolist(),
            "cbl": np.asarray(self.cbl, dtype=float).tolist(),
            "power": float(self.power),
            "mean_sinr": self.mean_sinr,
            "sum_cbl": self.sum_cbl,
            "constraint_violations": {k: float(v) for k, v in self.constraint_violations.items()},
            "extra": {
                k: (float(v) if np.isscalar(v) else v)
                for k, v in self.extra.items()
            },
        }


@dataclass
class ExperimentPaths:
    """统一描述一次实验的输出目录结构。"""

    outputs_root: Path
    problem_name: str
    solver_name: str
    run_name: str
    solver_root: Path
    run_dir: Path
    tb_dir: Path
    ckpt_dir: Path
    plots_dir: Path
    train_csv: Path
    eval_csv: Path
    problem_config_json: Path
    solver_config_json: Path
    summary_json: Path
    evaluation_summary_json: Path

