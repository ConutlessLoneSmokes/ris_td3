from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from problems.ris_miso_urllc.config import ProblemConfig


def practical_ris_amplitude(theta: np.ndarray, beta_min: float, alpha_ris: float, phi_ris: float) -> np.ndarray:
    """根据论文中的实际 RIS 模型计算反射幅度。"""
    base = (np.sin(theta - phi_ris) + 1.0) / 2.0
    return (1.0 - beta_min) * np.power(base, alpha_ris) + beta_min


def ris_coefficients(theta: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """由相位向量生成实际复反射系数。"""
    beta = practical_ris_amplitude(theta, cfg.beta_min, cfg.alpha_ris, cfg.phi_ris)
    return beta * np.exp(1j * theta)


def build_theta_matrix(theta: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """将复反射系数向量写成对角矩阵。"""
    return np.diag(ris_coefficients(theta, cfg))


def cascaded_channel(h_ru_k: np.ndarray, h_br: np.ndarray) -> np.ndarray:
    """构造第 k 个用户的级联信道。"""
    return np.diag(np.conjugate(h_ru_k)) @ h_br


def effective_scalar(
    h_ru_k: np.ndarray,
    h_br: np.ndarray,
    theta: np.ndarray,
    w_k: np.ndarray,
    cfg: ProblemConfig,
) -> complex:
    """计算某个用户在给定 RIS 与波束赋形下的等效复信道标量。"""
    theta_vec = ris_coefficients(theta, cfg)
    h_tilde = cascaded_channel(h_ru_k, h_br)
    return complex(np.conjugate(theta_vec) @ h_tilde @ w_k)


def sinr_all(
    h_br: np.ndarray,
    h_ru: np.ndarray,
    theta: np.ndarray,
    beamforming: np.ndarray,
    cfg: ProblemConfig,
) -> np.ndarray:
    """计算所有用户的 SINR。"""
    out = np.zeros(cfg.K, dtype=float)
    theta_vec = ris_coefficients(theta, cfg)

    for k in range(cfg.K):
        h_tilde_k = cascaded_channel(h_ru[k], h_br)
        desired = np.abs(np.conjugate(theta_vec) @ h_tilde_k @ beamforming[k]) ** 2

        interference = 0.0
        for i in range(cfg.K):
            if i == k:
                continue
            interference += np.abs(np.conjugate(theta_vec) @ h_tilde_k @ beamforming[i]) ** 2

        out[k] = desired / (interference + cfg.sigma2 + cfg.eps_div)

    return out


def shannon_capacity(sinr: np.ndarray) -> np.ndarray:
    """计算 Shannon 容量项。"""
    return np.log2(1.0 + sinr)


def channel_dispersion(sinr: np.ndarray) -> np.ndarray:
    """计算有限块长中的信道离散度项。"""
    return (1.0 / (math.log(2.0) ** 2)) * (1.0 - 1.0 / ((1.0 + sinr) ** 2))


def fbl_bits(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> np.ndarray:
    """计算每个用户在有限块长条件下可传输的信息比特数。"""
    qinv = norm.isf(error_prob)
    capacity = shannon_capacity(sinr)
    dispersion = channel_dispersion(sinr)
    return cbl * capacity - qinv * np.sqrt(np.maximum(cbl * dispersion, 0.0))


def reward_total_fbl(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> float:
    """将所有用户的有限块长比特数求和作为 reward。"""
    return float(np.sum(fbl_bits(sinr, cbl, error_prob)))
