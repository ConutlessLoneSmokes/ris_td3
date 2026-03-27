import math

import numpy as np
from scipy.stats import norm

from configs.default import SystemConfig


def practical_ris_amplitude(theta: np.ndarray, beta_min: float, alpha_ris: float, phi_ris: float) -> np.ndarray:
    """根据论文中的非理想幅相耦合模型计算 RIS 幅度响应。"""
    base = (np.sin(theta - phi_ris) + 1.0) / 2.0
    return (1.0 - beta_min) * np.power(base, alpha_ris) + beta_min


def ris_coefficients(theta: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """将 RIS 相位向量转换为实际复反射系数。"""
    beta = practical_ris_amplitude(theta, cfg.beta_min, cfg.alpha_ris, cfg.phi_ris)
    return beta * np.exp(1j * theta)


def build_theta_matrix(theta: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """将 RIS 复反射系数向量转换为对角反射矩阵。"""
    coeff = ris_coefficients(theta, cfg)
    return np.diag(coeff)


def cascaded_channel(h_ru_k: np.ndarray, h_br: np.ndarray) -> np.ndarray:
    """构造第 k 个用户对应的级联信道矩阵。"""
    return np.diag(np.conjugate(h_ru_k)) @ h_br


def effective_scalar(
    h_ru_k: np.ndarray,
    h_br: np.ndarray,
    theta: np.ndarray,
    w_k: np.ndarray,
    cfg: SystemConfig,
) -> complex:
    """计算第 k 个用户在给定 RIS 和波束赋形下的等效复标量信道。"""
    theta_vec = ris_coefficients(theta, cfg)
    h_tilde = cascaded_channel(h_ru_k, h_br)
    return complex(np.conjugate(theta_vec) @ h_tilde @ w_k)


def sinr_all(h_br: np.ndarray, h_ru: np.ndarray, theta: np.ndarray, w: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """计算所有用户的 SINR。"""
    out = np.zeros(cfg.K, dtype=float)
    theta_vec = ris_coefficients(theta, cfg)

    for k in range(cfg.K):
        h_tilde_k = cascaded_channel(h_ru[k], h_br)
        # 目标信号功率由当前用户自己的波束赋形向量产生。
        desired = np.abs(np.conjugate(theta_vec) @ h_tilde_k @ w[k]) ** 2

        interf = 0.0
        for i in range(cfg.K):
            if i == k:
                continue
            # 其他用户的波束在当前用户处形成多用户干扰。
            interf += np.abs(np.conjugate(theta_vec) @ h_tilde_k @ w[i]) ** 2

        out[k] = desired / (interf + cfg.sigma2 + cfg.eps_div)

    return out


def shannon_capacity(sinr: np.ndarray) -> np.ndarray:
    """根据 Shannon 公式计算单位信道使用下的容量。"""
    return np.log2(1.0 + sinr)


def channel_dispersion(sinr: np.ndarray) -> np.ndarray:
    """计算有限块长通信中的信道离散度项。"""
    return (1.0 / (math.log(2.0) ** 2)) * (1.0 - 1.0 / ((1.0 + sinr) ** 2))


def fbl_bits(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> np.ndarray:
    """计算每个用户在有限块长条件下可传输的信息比特数。"""
    qinv = norm.isf(error_prob)
    c = shannon_capacity(sinr)
    v = channel_dispersion(sinr)
    # 分别对应容量主项、有限块长惩罚项以及对块长的校正项。
    return cbl * c - qinv * np.sqrt(np.maximum(cbl * v, 0.0))


def reward_total_fbl(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> float:
    """将所有用户的有限块长比特数求和，作为环境奖励。"""
    return float(np.sum(fbl_bits(sinr, cbl, error_prob)))
