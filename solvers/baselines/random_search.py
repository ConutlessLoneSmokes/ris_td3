from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from core.io import write_json
from core.registry import register_solver
from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.constraints import sample_random_solution
from problems.ris_miso_urllc.evaluator import Evaluator
from problems.ris_miso_urllc.types import ProblemInstance, Solution
from solvers.base import Solver


@dataclass
class RandomSearchConfig:
    """随机搜索基线配置。"""

    num_candidates: int = 256
    seed_offset: int = 1000


class RandomSearchSolver(Solver):
    """在可行域中随机采样并选取最佳解。"""

    name = "random_search"

    def __init__(self, problem_config: ProblemConfig, solver_config: RandomSearchConfig):
        super().__init__(problem_config, solver_config)
        self.evaluator = Evaluator(problem_config)
        self.rng = torch.Generator().manual_seed(problem_config.seed + solver_config.seed_offset)

    def _numpy_rng(self, instance_index: int) -> "np.random.Generator":
        import numpy as np

        seed = int(self.problem_config.seed + self.solver_config.seed_offset + instance_index)
        return np.random.default_rng(seed)

    def solve(self, instance: ProblemInstance) -> Solution:
        import numpy as np

        rng = self._numpy_rng(int(np.sum(np.abs(instance.h_br)) * 1e6) % 100000)
        best_solution: Solution | None = None
        best_reward = float("-inf")

        for _ in range(self.solver_config.num_candidates):
            solution = sample_random_solution(self.problem_config, rng)
            metrics = self.evaluator.evaluate(instance, solution)
            if metrics.reward > best_reward:
                best_reward = metrics.reward
                best_solution = solution

        assert best_solution is not None
        return best_solution

    def save(self, path: str | Path) -> None:
        write_json(Path(path), {
            "problem_name": self.problem_config.name,
            "solver_name": self.name,
            "solver_config": vars(self.solver_config),
        })

    def load(self, path: str | Path) -> None:
        del path


register_solver("random_search", "deterministic", RandomSearchConfig, RandomSearchSolver)
