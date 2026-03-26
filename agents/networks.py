import torch
from torch import nn


def _linear_block(in_dim: int, out_dim: int, use_layer_norm: bool, apply_activation: bool = True) -> nn.Sequential:
    """构造一个线性层块，可选 LayerNorm 和 ReLU。"""
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    if apply_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """TD3 的策略网络，将状态映射为 [-1, 1] 范围内的连续动作。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...], use_layer_norm: bool):
        """根据配置的隐藏层宽度构造多层感知机。"""
        super().__init__()
        dims = (state_dim,) + tuple(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(_linear_block(in_dim, out_dim, use_layer_norm))
        # 最后一层使用 tanh，保证动作天然落在约束映射所需的区间内。
        layers.extend([nn.Linear(dims[-1], action_dim), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """执行前向传播，输出连续动作。"""
        return self.net(state)


class Critic(nn.Module):
    """TD3 的双 Q 网络，用于减轻动作价值高估问题。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_hidden_dim: int,
        action_hidden_dim: int,
        hidden_dims: tuple[int, ...],
        use_layer_norm: bool,
    ):
        """分别构造 Q1 和 Q2 的状态分支、动作分支与融合头部。"""
        super().__init__()
        self.q1_state = _linear_block(state_dim, state_hidden_dim, use_layer_norm)
        self.q1_action = _linear_block(action_dim, action_hidden_dim, use_layer_norm)
        self.q1_head = self._build_head(state_hidden_dim, hidden_dims, use_layer_norm)

        self.q2_state = _linear_block(state_dim, state_hidden_dim, use_layer_norm)
        self.q2_action = _linear_block(action_dim, action_hidden_dim, use_layer_norm)
        self.q2_head = self._build_head(state_hidden_dim, hidden_dims, use_layer_norm)

    @staticmethod
    def _build_head(input_dim: int, hidden_dims: tuple[int, ...], use_layer_norm: bool) -> nn.Sequential:
        """构造状态与动作特征融合后的回归头。"""
        dims = (input_dim,) + tuple(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(_linear_block(in_dim, out_dim, use_layer_norm))
        layers.append(nn.Linear(dims[-1], 1))
        return nn.Sequential(*layers)

    def _forward_single(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_branch: nn.Module,
        action_branch: nn.Module,
        head: nn.Module,
    ) -> torch.Tensor:
        """执行单个 Q 网络的前向传播。"""
        # 先分别抽取状态特征和动作特征，再相加融合。
        fused = state_branch(state) + action_branch(action)
        return head(fused)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """同时返回 Q1 与 Q2，供 TD3 取最小值使用。"""
        q1 = self._forward_single(state, action, self.q1_state, self.q1_action, self.q1_head)
        q2 = self._forward_single(state, action, self.q2_state, self.q2_action, self.q2_head)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """仅计算 Q1，用于 Actor 更新。"""
        return self._forward_single(state, action, self.q1_state, self.q1_action, self.q1_head)
