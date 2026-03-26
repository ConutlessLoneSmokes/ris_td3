import numpy as np
import torch


class ReplayBuffer:
    """固定容量的环形经验回放缓存。"""

    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        """预分配状态、动作、奖励和终止标记的存储空间。"""
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """写入一条经验，并自动覆盖最旧样本。"""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr] = np.asarray([reward], dtype=np.float32)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.ptr] = np.asarray([float(done)], dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """随机采样一个 batch，并直接转换到目标设备。"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        """返回当前缓存中可用样本数。"""
        return self.size
