"""求解器集合与注册入口。"""

from solvers.baselines.random_search import RandomSearchConfig, RandomSearchSolver
from solvers.rl.ddpg import DDPGConfig, DDPGSolver
from solvers.rl.td3 import TD3Config, TD3Solver

__all__ = [
    "TD3Config",
    "TD3Solver",
    "DDPGConfig",
    "DDPGSolver",
    "RandomSearchConfig",
    "RandomSearchSolver",
]
