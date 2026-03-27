from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.networks import Actor, Critic
from agents.replay_buffer import ReplayBuffer
from configs.default import SystemConfig


class TD3Agent:
    """适配当前 RIS 环境的 TD3 智能体实现。"""

    def __init__(self, state_dim: int, action_dim: int, cfg: SystemConfig):
        """构造主网络、目标网络和优化器。"""
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = self._resolve_device(cfg.device)

        self.actor = Actor(
            state_dim,
            action_dim,
            cfg.actor_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.actor_target = Actor(
            state_dim,
            action_dim,
            cfg.actor_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            state_dim,
            action_dim,
            cfg.critic_state_hidden_dim,
            cfg.critic_action_hidden_dim,
            cfg.critic_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target = Critic(
            state_dim,
            action_dim,
            cfg.critic_state_hidden_dim,
            cfg.critic_action_hidden_dim,
            cfg.critic_hidden_dims,
            cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.total_it = 0

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        """根据配置和硬件条件选择实际计算设备。"""
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据当前策略网络输出动作。"""
        del deterministic
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action.astype(np.float32), -1.0, 1.0)

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float] | None:
        """执行一次 TD3 参数更新。"""
        if len(replay_buffer) < self.cfg.batch_size:
            return None

        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(self.cfg.batch_size, self.device)

        with torch.no_grad():
            # 为目标动作添加截断高斯噪声，实现 target policy smoothing。
            noise = torch.randn_like(action) * self.cfg.policy_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 核心修改 2：加入梯度裁剪，限制最大梯度范数为 1.0 或 10.0
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=50.0)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        actor_updated = False

        if self.total_it % self.cfg.policy_delay == 0:
            # Actor 的目标是最大化 Q1，因此优化时取负号。
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 核心修改 2：加入梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()

            # 只有在 Actor 更新时，同步对目标网络做软更新。
            self._soft_update(self.actor_target, self.actor, self.cfg.tau)
            self._soft_update(self.critic_target, self.critic, self.cfg.tau)

            actor_loss_value = float(actor_loss.item())
            actor_updated = True

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "actor_updated": float(actor_updated),
        }

    @staticmethod
    def _soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
        """按 Polyak averaging 方式更新目标网络。"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str | Path) -> None:
        """保存网络参数、优化器状态和训练步数。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "total_it": self.total_it,
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
        """从 checkpoint 恢复网络参数和优化器状态。"""
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.total_it = int(checkpoint["total_it"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
