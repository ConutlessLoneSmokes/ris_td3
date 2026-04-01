from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from problems.ris_miso_urllc.config import ProblemConfig


def db_to_linear(db: float) -> float:
    """将 dB 转成线性刻度。"""
    return 10 ** (db / 10.0)


def complex_gaussian(
    shape,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成复高斯随机变量。"""
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return scale * (real + 1j * imag) / math.sqrt(2.0)


def row_vec(mat: np.ndarray) -> np.ndarray:
    """按行优先方式展平矩阵。"""
    return mat.reshape(-1, order="C")


def upa_response(
    azimuth: float,
    elevation: float,
    n1: int,
    n2: int,
    d: float,
    wavelength: float,
) -> np.ndarray:
    """生成 UPA 阵列响应向量。"""
    response = np.zeros((n1, n2), dtype=np.complex128)
    for i in range(n1):
        for j in range(n2):
            phase = (
                2.0
                * math.pi
                * d
                / wavelength
                * (i * math.cos(azimuth) + j * math.sin(azimuth))
                * math.sin(elevation)
            )
            response[i, j] = np.exp(1j * phase)
    return row_vec(response)


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两点间三维欧氏距离。"""
    return float(np.linalg.norm(p1 - p2))


def pathloss_linear(distance_m: float, pl0_db: float, pathloss_exp: float) -> float:
    """按对数距离模型计算线性路损系数。"""
    pl_db = pl0_db - 10.0 * pathloss_exp * math.log10(max(distance_m, 1e-6))
    return db_to_linear(pl_db)


def az_el_from_points(tx: np.ndarray, rx: np.ndarray) -> Tuple[float, float]:
    """根据收发坐标计算方位角与俯仰角。"""
    vec = rx - tx
    dx, dy, dz = vec
    azimuth = math.atan2(dy, dx)
    horizontal = math.sqrt(dx * dx + dy * dy)
    elevation = math.atan2(dz, max(horizontal, 1e-12))
    return azimuth, elevation


class ChannelGenerator:
    """按论文场景生成 BS-RIS 和 RIS-用户信道。"""

    def __init__(self, cfg: ProblemConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)

    def bs_xyz(self) -> np.ndarray:
        """返回基站三维坐标。"""
        return np.array([self.cfg.bs_pos[0], self.cfg.bs_pos[1], self.cfg.bs_height], dtype=float)

    def ris_xyz(self) -> np.ndarray:
        """返回 RIS 三维坐标。"""
        return np.array([self.cfg.ris_pos[0], self.cfg.ris_pos[1], self.cfg.ris_height], dtype=float)

    def user_xyz(self, k: int) -> np.ndarray:
        """返回第 k 个用户的三维坐标。"""
        return np.array(
            [self.cfg.user_pos[k][0], self.cfg.user_pos[k][1], self.cfg.user_height],
            dtype=float,
        )

    def generate_bs_ris(self) -> np.ndarray:
        """生成 BS 到 RIS 的 Rician 信道。"""
        cfg = self.cfg
        bs = self.bs_xyz()
        ris = self.ris_xyz()

        d_br = distance_3d(bs, ris)
        beta_br = pathloss_linear(d_br, cfg.pl0_db, cfg.pathloss_exp)

        az_ris, el_ris = az_el_from_points(bs, ris)
        az_bs, el_bs = az_el_from_points(ris, bs)

        a_ris = upa_response(az_ris, el_ris, cfg.Nx, cfg.Ny, cfg.element_spacing, cfg.wavelength)
        a_bs = upa_response(az_bs, el_bs, cfg.Mx, cfg.My, cfg.element_spacing, cfg.wavelength)

        h_los = math.sqrt(beta_br) * np.outer(np.conjugate(a_ris), a_bs)
        h_nlos = complex_gaussian((cfg.N, cfg.M), scale=math.sqrt(beta_br), rng=self.rng)

        zeta = cfg.zeta_br
        return (
            math.sqrt(zeta / (zeta + 1.0)) * h_los
            + math.sqrt(1.0 / (zeta + 1.0)) * h_nlos
        ).astype(np.complex128)

    def generate_ris_users(self) -> np.ndarray:
        """生成 RIS 到所有用户的 Rician 信道。"""
        cfg = self.cfg
        ris = self.ris_xyz()
        out = np.zeros((cfg.K, cfg.N), dtype=np.complex128)

        for k in range(cfg.K):
            user = self.user_xyz(k)
            d_ru = distance_3d(ris, user)
            beta_ru = pathloss_linear(d_ru, cfg.pl0_db, cfg.pathloss_exp)

            az_ru, el_ru = az_el_from_points(ris, user)
            a_ru = upa_response(az_ru, el_ru, cfg.Nx, cfg.Ny, cfg.element_spacing, cfg.wavelength)

            h_los = math.sqrt(beta_ru) * a_ru
            h_nlos = complex_gaussian((cfg.N,), scale=math.sqrt(beta_ru), rng=self.rng)

            zeta = cfg.zeta_ru
            out[k] = (
                math.sqrt(zeta / (zeta + 1.0)) * h_los
                + math.sqrt(1.0 / (zeta + 1.0)) * h_nlos
            )

        return out
