from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.types import ProblemInstance, Solution


class Solver(ABC):
    """统一求解器抽象。"""

    name: str = "solver"

    def __init__(self, problem_config: ProblemConfig, solver_config: Any):
        self.problem_config = problem_config
        self.solver_config = solver_config

    def setup(self, problem_config: ProblemConfig) -> None:
        """在需要时更新问题配置。"""
        self.problem_config = problem_config

    @abstractmethod
    def solve(self, instance: ProblemInstance) -> Solution:
        """对单个问题实例给出候选解。"""

    def save(self, path: str | Path) -> None:
        """保存求解器状态。默认无状态。"""
        del path

    def load(self, path: str | Path) -> None:
        """加载求解器状态。默认无状态。"""
        del path
