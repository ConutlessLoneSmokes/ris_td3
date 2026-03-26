# envs/fbl.py
import math
import numpy as np
from scipy.stats import norm
from configs.default import SystemConfig


def practical_ris_amplitude(theta: np.ndarray, beta_min: float, alpha_ris: float, phi_ris: float) -> np.ndarray:
    base = (np.sin(theta - phi_ris) + 1.0) / 2.0
    return (1.0 - beta_min) * np.power(base, alpha_ris) + beta_min


def ris_coefficients(theta: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    beta = practical_ris_amplitude(theta, cfg.beta_min, cfg.alpha_ris, cfg.phi_ris)
    return beta * np.exp(1j * theta)


def build_theta_matrix(theta: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    coeff = ris_coefficients(theta, cfg)
    return np.diag(coeff)


def cascaded_channel(h_ru_k: np.ndarray, h_br: np.ndarray) -> np.ndarray:
    return np.diag(np.conjugate(h_ru_k)) @ h_br


def effective_scalar(h_ru_k: np.ndarray, h_br: np.ndarray, theta: np.ndarray, w_k: np.ndarray, cfg: SystemConfig) -> complex:
    theta_vec = ris_coefficients(theta, cfg)
    h_tilde = cascaded_channel(h_ru_k, h_br)
    return np.conjugate(theta_vec) @ h_tilde @ w_k


def sinr_all(h_br: np.ndarray, h_ru: np.ndarray, theta: np.ndarray, w: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    out = np.zeros(cfg.K, dtype=float)
    theta_vec = ris_coefficients(theta, cfg)

    for k in range(cfg.K):
        h_tilde_k = cascaded_channel(h_ru[k], h_br)
        desired = np.abs(np.conjugate(theta_vec) @ h_tilde_k @ w[k]) ** 2

        interf = 0.0
        for i in range(cfg.K):
            if i == k:
                continue
            interf += np.abs(np.conjugate(theta_vec) @ h_tilde_k @ w[i]) ** 2

        out[k] = desired / (interf + cfg.sigma2 + cfg.eps_div)

    return out


def shannon_capacity(sinr: np.ndarray) -> np.ndarray:
    return np.log2(1.0 + sinr)


def channel_dispersion(sinr: np.ndarray) -> np.ndarray:
    return (1.0 / (math.log(2.0) ** 2)) * (1.0 - 1.0 / ((1.0 + sinr) ** 2))


def fbl_bits(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> np.ndarray:
    qinv = norm.isf(error_prob)
    c = shannon_capacity(sinr)
    v = channel_dispersion(sinr)
    return cbl * c - qinv * np.sqrt(np.maximum(cbl * v, 0.0)) + np.log2(np.maximum(cbl, 1e-12))


def reward_total_fbl(sinr: np.ndarray, cbl: np.ndarray, error_prob: float) -> float:
    return float(np.sum(fbl_bits(sinr, cbl, error_prob)))