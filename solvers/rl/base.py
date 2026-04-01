from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.rl_env import RISEnv
from problems.ris_miso_urllc.types import ProblemInstance, Solution
from solvers.base import Solver


@dataclass
class RLBaseConfig:
    """RL 方法共享训练配置。"""

    actor_hidden_dims: tuple[int, int, int] = (800, 400, 200)
    critic_state_hidden_dim: int = 800
    critic_action_hidden_dim: int = 800
    critic_hidden_dims: tuple[int, int] = (600, 400)
    use_layer_norm: bool = True

    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    exploration_noise: float = float(np.sqrt(0.1))

    buffer_size: int = 10000
    batch_size: int = 64
    warmup_episodes: int = 0
    reward_scale: float = 0.01
    device: str = "cuda"

    actor_grad_clip: float = 10.0
    critic_grad_clip: float = 50.0


class RLSolver(Solver):
    """RL 求解器基类。"""

    def __init__(self, problem_config: ProblemConfig, solver_config: RLBaseConfig):
        super().__init__(problem_config, solver_config)
        self.device = self._resolve_device(solver_config.device)
        self.state_dim = 0
        self.action_dim = 0

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        """根据配置与硬件可用性决定实际设备。"""
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def bind_environment(self, state_dim: int, action_dim: int) -> None:
        """在拿到环境维度后构建网络。"""
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._build_networks()

    @abstractmethod
    def _build_networks(self) -> None:
        """由子类完成网络与优化器构建。"""

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据当前策略输出动作。"""

    @abstractmethod
    def update(self, replay_buffer) -> dict[str, float] | None:
        """执行一次参数更新。"""

    def solve(self, instance: ProblemInstance) -> Solution:
        """在固定场景下滚动多步决策，返回最终解。"""
        env = RISEnv(self.problem_config)
        state = env.reset(instance)
        last_solution = env.last_solution
        for _ in range(self.problem_config.max_steps):
            action = self.select_action(state, deterministic=True)
            state, _, done, info = env.step(action)
            last_solution = info["solution"]
            if done:
                break
        assert last_solution is not None
        return last_solution

    def save(self, path: str | Path) -> None:
        """由子类实现。"""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """由子类实现。"""
        raise NotImplementedError
