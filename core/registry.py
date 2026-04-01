from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ProblemRegistration:
    """问题注册项。"""

    name: str
    config_factory: Callable[[], Any]
    builder: Callable[[Any], dict[str, Any]]


@dataclass
class SolverRegistration:
    """求解器注册项。"""

    name: str
    category: str
    config_factory: Callable[[], Any]
    builder: Callable[[Any], Any]


PROBLEM_REGISTRY: dict[str, ProblemRegistration] = {}
SOLVER_REGISTRY: dict[str, SolverRegistration] = {}


def register_problem(
    name: str,
    config_factory: Callable[[], Any],
    builder: Callable[[Any], dict[str, Any]],
) -> None:
    """注册一个问题。"""
    PROBLEM_REGISTRY[name] = ProblemRegistration(
        name=name,
        config_factory=config_factory,
        builder=builder,
    )


def register_solver(
    name: str,
    category: str,
    config_factory: Callable[[], Any],
    builder: Callable[[Any], Any],
) -> None:
    """注册一个求解器。"""
    SOLVER_REGISTRY[name] = SolverRegistration(
        name=name,
        category=category,
        config_factory=config_factory,
        builder=builder,
    )


def get_problem_registration(name: str) -> ProblemRegistration:
    """读取问题注册项。"""
    if name not in PROBLEM_REGISTRY:
        raise KeyError(f"未注册的问题: {name}")
    return PROBLEM_REGISTRY[name]


def get_solver_registration(name: str) -> SolverRegistration:
    """读取求解器注册项。"""
    if name not in SOLVER_REGISTRY:
        raise KeyError(f"未注册的求解器: {name}")
    return SOLVER_REGISTRY[name]
