from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProblemInstance:
    """一次固定信道 realization 下的优化问题实例。"""

    h_br: np.ndarray
    h_ru: np.ndarray
    bs_xyz: np.ndarray
    ris_xyz: np.ndarray
    user_xyz: np.ndarray
    sigma2: float
    p_total_watt: float


@dataclass
class Solution:
    """问题层统一候选解表示。"""

    theta: np.ndarray
    beamforming: np.ndarray
    cbl: np.ndarray
