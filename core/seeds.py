from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """统一设置 Python、NumPy 与 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

