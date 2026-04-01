from __future__ import annotations

# 导入内置问题与求解器，确保注册器被填充。
import problems  # noqa: F401
import solvers  # noqa: F401

from core.registry import PROBLEM_REGISTRY, SOLVER_REGISTRY, get_problem_registration, get_solver_registration
from problems.ris_miso_urllc.config import ProblemConfig
from solvers.baselines.random_search import RandomSearchConfig
from solvers.rl.ddpg import DDPGConfig
from solvers.rl.td3 import TD3Config

DEFAULT_PROBLEM_NAME = "ris_miso_urllc"
DEFAULT_SOLVER_NAME = "td3"


def build_problem_config(problem_name: str = DEFAULT_PROBLEM_NAME):
    """构造某个问题的默认配置。"""
    return get_problem_registration(problem_name).config_factory()


def build_solver_config(solver_name: str = DEFAULT_SOLVER_NAME):
    """构造某个求解器的默认配置。"""
    return get_solver_registration(solver_name).config_factory()


def list_registered_problems() -> list[str]:
    """返回已注册问题名称。"""
    return sorted(PROBLEM_REGISTRY)


def list_registered_solvers() -> list[str]:
    """返回已注册求解器名称。"""
    return sorted(SOLVER_REGISTRY)

__all__ = [
    "DEFAULT_PROBLEM_NAME",
    "DEFAULT_SOLVER_NAME",
    "ProblemConfig",
    "TD3Config",
    "DDPGConfig",
    "RandomSearchConfig",
    "build_problem_config",
    "build_solver_config",
    "list_registered_problems",
    "list_registered_solvers",
]
