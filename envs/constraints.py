# envs/constraints.py
import numpy as np
from configs.default import SystemConfig


def map_raw_theta(raw_theta: np.ndarray) -> np.ndarray:
    raw_theta = np.clip(raw_theta, -1.0, 1.0)
    return np.pi * raw_theta


def map_raw_cbl(raw_cbl: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    raw_cbl = np.clip(raw_cbl, -1.0, 1.0)
    a_tilde = (raw_cbl + 1.0) / 2.0

    c_min = np.full(cfg.K, cfg.min_cbl, dtype=float)
    numerator = cfg.total_cbl - np.sum(c_min)
    denominator = np.sum(a_tilde) + cfg.eps_div

    c = numerator / denominator * a_tilde + c_min
    return c


def map_raw_beamforming(raw_mag: np.ndarray, raw_phase: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    raw_mag: [K, M] in [-1, 1]
    raw_phase: [K, M] in [-1, 1]
    return: complex beamforming matrix [K, M]
    """
    raw_mag = np.clip(raw_mag, -1.0, 1.0)
    raw_phase = np.clip(raw_phase, -1.0, 1.0)

    mag = (raw_mag + 1.0) / 2.0
    phase = np.pi * raw_phase
    w = mag * np.exp(1j * phase)

    power = np.sum(np.abs(w) ** 2)
    if power <= cfg.eps_div:
        w = np.ones((cfg.K, cfg.M), dtype=np.complex128)
        power = np.sum(np.abs(w) ** 2)

    w = w * np.sqrt(cfg.p_total_watt / power)
    return w.astype(np.complex128)