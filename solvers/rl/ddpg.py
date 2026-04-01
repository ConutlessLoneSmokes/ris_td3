from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.registry import register_solver
from problems.ris_miso_urllc.config import ProblemConfig
from solvers.rl.base import RLBaseConfig, RLSolver
from solvers.rl.networks import Actor, SingleCritic


@dataclass
class DDPGConfig(RLBaseConfig):
    """DDPG 专属配置。"""

    pass


class DDPGSolver(RLSolver):
    """适配当前 RIS 问题的 DDPG 求解器。"""

    name = "ddpg"

    def __init__(self, problem_config: ProblemConfig, solver_config: DDPGConfig):
        super().__init__(problem_config, solver_config)
        self.total_it = 0

    @property
    def cfg(self) -> DDPGConfig:
        return self.solver_config

    def _build_networks(self) -> None:
        cfg = self.cfg
        self.actor = Actor(self.state_dim, self.action_dim, cfg.actor_hidden_dims, cfg.use_layer_norm).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, cfg.actor_hidden_dims, cfg.use_layer_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = SingleCritic(
            self.state_dim,
            self.action_dim,
            cfg.critic_state_hidden_dim,
            cfg.critic_action_hidden_dim,
            cfg.critic_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target = SingleCritic(
            self.state_dim,
            self.action_dim,
            cfg.critic_state_hidden_dim,
            cfg.critic_action_hidden_dim,
            cfg.critic_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del deterministic
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action.astype(np.float32), -1.0, 1.0)

    def update(self, replay_buffer) -> dict[str, float] | None:
        if len(replay_buffer) < self.cfg.batch_size:
            return None

        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(self.cfg.batch_size, self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.cfg.critic_grad_clip)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.cfg.actor_grad_clip)
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor, self.cfg.tau)
        self._soft_update(self.critic_target, self.critic, self.cfg.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "actor_updated": 1.0,
        }

    @staticmethod
    def _soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "problem_name": self.problem_config.name,
                "solver_name": self.name,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "total_it": self.total_it,
                "solver_config": vars(self.cfg),
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.total_it = int(checkpoint["total_it"])
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self._build_networks()
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


register_solver("ddpg", "rl", DDPGConfig, DDPGSolver)
