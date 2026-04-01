from __future__ import annotations

import numpy as np

from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.types import Solution


def map_raw_theta(raw_theta: np.ndarray) -> np.ndarray:
    """将 [-1, 1] 归一化动作映射到 [-pi, pi]。"""
    return np.pi * np.clip(raw_theta, -1.0, 1.0)


def map_raw_cbl(raw_cbl: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """将归一化动作映射成满足总块长和最小块长的 CBL。"""
    raw_cbl = np.clip(raw_cbl, -1.0, 1.0)
    a_tilde = (raw_cbl + 1.0) / 2.0

    c_min = np.full(cfg.K, cfg.min_cbl, dtype=float)
    numerator = cfg.total_cbl - np.sum(c_min)
    denominator = np.sum(a_tilde) + cfg.eps_div
    return numerator / denominator * a_tilde + c_min


def map_raw_beamforming(raw_mag: np.ndarray, raw_phase: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """将归一化幅度和相位映射成满足总功率约束的复波束赋形矩阵。"""
    raw_mag = np.clip(raw_mag, -1.0, 1.0)
    raw_phase = np.clip(raw_phase, -1.0, 1.0)

    mag = (raw_mag + 1.0) / 2.0
    phase = np.pi * raw_phase
    beamforming = mag * np.exp(1j * phase)

    power = np.sum(np.abs(beamforming) ** 2)
    if power <= cfg.eps_div:
        beamforming = np.ones((cfg.K, cfg.M), dtype=np.complex128)
        power = np.sum(np.abs(beamforming) ** 2)

    beamforming = beamforming * np.sqrt(cfg.p_total_watt / power)
    return beamforming.astype(np.complex128)


def normalize_beamforming(beamforming: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """对任意复波束赋形矩阵做总功率归一化。"""
    power = np.sum(np.abs(beamforming) ** 2)
    if power <= cfg.eps_div:
        beamforming = np.ones((cfg.K, cfg.M), dtype=np.complex128)
        power = np.sum(np.abs(beamforming) ** 2)
    return beamforming * np.sqrt(cfg.p_total_watt / power)


def sample_random_solution(cfg: ProblemConfig, rng: np.random.Generator) -> Solution:
    """直接在物理变量空间中采样一个随机可行解。"""
    theta = rng.uniform(-np.pi, np.pi, size=cfg.N)

    weights = rng.uniform(0.0, 1.0, size=cfg.K)
    weights = weights / (np.sum(weights) + cfg.eps_div)
    extra_cbl = max(cfg.total_cbl - cfg.K * cfg.min_cbl, 0.0)
    cbl = np.full(cfg.K, cfg.min_cbl, dtype=float) + extra_cbl * weights

    raw_real = rng.standard_normal((cfg.K, cfg.M))
    raw_imag = rng.standard_normal((cfg.K, cfg.M))
    beamforming = normalize_beamforming(raw_real + 1j * raw_imag, cfg)

    return Solution(
        theta=theta.astype(np.float64),
        beamforming=beamforming.astype(np.complex128),
        cbl=cbl.astype(np.float64),
    )


def compute_constraint_violations(solution: Solution, cfg: ProblemConfig) -> dict[str, float]:
    """计算各类基础物理约束的违背量。"""
    power = float(np.sum(np.abs(solution.beamforming) ** 2))
    return {
        "power": max(0.0, power - cfg.p_total_watt),
        "cbl_sum": abs(float(np.sum(solution.cbl)) - cfg.total_cbl),
        "cbl_min": max(0.0, cfg.min_cbl - float(np.min(solution.cbl))),
    }
