import torch
from torch import nn


def _linear_block(in_dim: int, out_dim: int, use_layer_norm: bool, apply_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    if apply_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """TD3 策略网络，输出范围约束在 [-1, 1]。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...], use_layer_norm: bool):
        super().__init__()
        dims = (state_dim,) + tuple(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(_linear_block(in_dim, out_dim, use_layer_norm))
        layers.extend([nn.Linear(dims[-1], action_dim), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """双 Q 网络，用于缓解 Q 值高估。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_hidden_dim: int,
        action_hidden_dim: int,
        hidden_dims: tuple[int, ...],
        use_layer_norm: bool,
    ):
        super().__init__()
        self.q1_state = _linear_block(state_dim, state_hidden_dim, use_layer_norm)
        self.q1_action = _linear_block(action_dim, action_hidden_dim, use_layer_norm)
        self.q1_head = self._build_head(state_hidden_dim, hidden_dims, use_layer_norm)

        self.q2_state = _linear_block(state_dim, state_hidden_dim, use_layer_norm)
        self.q2_action = _linear_block(action_dim, action_hidden_dim, use_layer_norm)
        self.q2_head = self._build_head(state_hidden_dim, hidden_dims, use_layer_norm)

    @staticmethod
    def _build_head(input_dim: int, hidden_dims: tuple[int, ...], use_layer_norm: bool) -> nn.Sequential:
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
        fused = state_branch(state) + action_branch(action)
        return head(fused)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self._forward_single(state, action, self.q1_state, self.q1_action, self.q1_head)
        q2 = self._forward_single(state, action, self.q2_state, self.q2_action, self.q2_head)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._forward_single(state, action, self.q1_state, self.q1_action, self.q1_head)
