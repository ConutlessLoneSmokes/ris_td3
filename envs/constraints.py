import numpy as np

from configs.default import SystemConfig


def map_raw_theta(raw_theta: np.ndarray) -> np.ndarray:
    """将 Actor 输出的归一化 RIS 相位映射到 [-pi, pi]。"""
    raw_theta = np.clip(raw_theta, -1.0, 1.0)
    return np.pi * raw_theta


def map_raw_cbl(raw_cbl: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """将归一化 CBL 动作映射为满足总块长和最小块长约束的实际值。"""
    raw_cbl = np.clip(raw_cbl, -1.0, 1.0)
    a_tilde = (raw_cbl + 1.0) / 2.0

    c_min = np.full(cfg.K, cfg.min_cbl, dtype=float)
    numerator = cfg.total_cbl - np.sum(c_min)
    denominator = np.sum(a_tilde) + cfg.eps_div

    # 先按比例分配剩余块长，再加上每个用户的最小块长下界。
    c = numerator / denominator * a_tilde + c_min
    return c


def map_raw_beamforming(raw_mag: np.ndarray, raw_phase: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """将幅度和相位动作映射为满足总功率约束的复数波束赋形矩阵。"""
    raw_mag = np.clip(raw_mag, -1.0, 1.0)
    raw_phase = np.clip(raw_phase, -1.0, 1.0)

    # 幅度从 [-1, 1] 线性映射到 [0, 1]，相位映射到 [-pi, pi]。
    mag = (raw_mag + 1.0) / 2.0
    phase = np.pi * raw_phase
    w = mag * np.exp(1j * phase)

    power = np.sum(np.abs(w) ** 2)
    if power <= cfg.eps_div:
        # 当网络输出几乎全零时，使用均匀非零初始化防止除零。
        w = np.ones((cfg.K, cfg.M), dtype=np.complex128)
        power = np.sum(np.abs(w) ** 2)

    # 最后统一缩放到总发射功率约束之内。
    w = w * np.sqrt(cfg.p_total_watt / power)
    return w.astype(np.complex128)
