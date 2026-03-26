import math
from typing import Tuple

import numpy as np

from configs.default import SystemConfig


def db_to_linear(db: float) -> float:
    """将 dB 标度转换为线性标度。"""
    return 10 ** (db / 10.0)


def complex_gaussian(shape, scale: float = 1.0, rng: np.random.Generator | None = None):
    """生成复高斯随机变量，实部与虚部独立同分布。"""
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return scale * (real + 1j * imag) / math.sqrt(2.0)


def row_vec(mat: np.ndarray) -> np.ndarray:
    """按行优先方式将矩阵展平为向量。"""
    return mat.reshape(-1, order="C")


def upa_response(azimuth: float, elevation: float, n1: int, n2: int, d: float, wavelength: float) -> np.ndarray:
    """生成 UPA 阵列响应向量。"""
    h = np.zeros((n1, n2), dtype=np.complex128)
    for i in range(n1):
        for j in range(n2):
            # 相位项由阵元位置、方位角、俯仰角和间距共同决定。
            phase = (
                2.0
                * math.pi
                * d
                / wavelength
                * (i * math.cos(azimuth) + j * math.sin(azimuth))
                * math.sin(elevation)
            )
            h[i, j] = np.exp(1j * phase)
    return row_vec(h)


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算三维欧氏距离。"""
    return float(np.linalg.norm(p1 - p2))


def pathloss_linear(distance_m: float, pl0_db: float, pathloss_exp: float) -> float:
    """根据对数距离路损模型计算线性路损系数。"""
    pl_db = pl0_db - 10.0 * pathloss_exp * math.log10(max(distance_m, 1e-6))
    return db_to_linear(pl_db)


def az_el_from_points(tx: np.ndarray, rx: np.ndarray) -> Tuple[float, float]:
    """根据收发点坐标计算方位角和俯仰角。"""
    vec = rx - tx
    dx, dy, dz = vec
    az = math.atan2(dy, dx)
    horiz = math.sqrt(dx * dx + dy * dy)
    el = math.atan2(dz, max(horiz, 1e-12))
    return az, el


class ChannelGenerator:
    """按照论文场景生成 BS-RIS 和 RIS-用户信道。"""

    def __init__(self, cfg: SystemConfig):
        """保存配置并初始化随机数发生器。"""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _bs_xyz(self) -> np.ndarray:
        """返回基站的三维坐标。"""
        return np.array([self.cfg.bs_pos[0], self.cfg.bs_pos[1], self.cfg.bs_height], dtype=float)

    def _ris_xyz(self) -> np.ndarray:
        """返回 RIS 的三维坐标。"""
        return np.array([self.cfg.ris_pos[0], self.cfg.ris_pos[1], self.cfg.ris_height], dtype=float)

    def _user_xyz(self, k: int) -> np.ndarray:
        """返回第 k 个用户的三维坐标。"""
        return np.array(
            [self.cfg.user_pos[k][0], self.cfg.user_pos[k][1], self.cfg.user_height],
            dtype=float,
        )

    def generate_bs_ris(self) -> np.ndarray:
        """生成 BS 到 RIS 的 Rician 信道矩阵。"""
        cfg = self.cfg
        bs = self._bs_xyz()
        ris = self._ris_xyz()

        d_br = distance_3d(bs, ris)
        beta_br = pathloss_linear(d_br, cfg.pl0_db, cfg.pathloss_exp)

        az_ris, el_ris = az_el_from_points(bs, ris)
        az_bs, el_bs = az_el_from_points(ris, bs)

        a_ris = upa_response(
            azimuth=az_ris,
            elevation=el_ris,
            n1=cfg.Nx,
            n2=cfg.Ny,
            d=cfg.element_spacing,
            wavelength=cfg.wavelength,
        )
        a_bs = upa_response(
            azimuth=az_bs,
            elevation=el_bs,
            n1=cfg.Mx,
            n2=cfg.My,
            d=cfg.element_spacing,
            wavelength=cfg.wavelength,
        )

        # LoS 分量由阵列响应外积构成，NLoS 分量由复高斯随机变量构成。
        h_los = math.sqrt(beta_br) * np.outer(np.conjugate(a_ris), a_bs)
        h_nlos = complex_gaussian((cfg.N, cfg.M), scale=math.sqrt(beta_br), rng=self.rng)

        zeta = cfg.zeta_br
        h = math.sqrt(zeta / (zeta + 1.0)) * h_los + math.sqrt(1.0 / (zeta + 1.0)) * h_nlos
        return h.astype(np.complex128)

    def generate_ris_users(self) -> np.ndarray:
        """生成 RIS 到所有用户的 Rician 信道向量。"""
        cfg = self.cfg
        ris = self._ris_xyz()
        out = np.zeros((cfg.K, cfg.N), dtype=np.complex128)

        for k in range(cfg.K):
            user = self._user_xyz(k)
            d_ru = distance_3d(ris, user)
            beta_ru = pathloss_linear(d_ru, cfg.pl0_db, cfg.pathloss_exp)

            az_ru, el_ru = az_el_from_points(ris, user)
            a_ru = upa_response(
                azimuth=az_ru,
                elevation=el_ru,
                n1=cfg.Nx,
                n2=cfg.Ny,
                d=cfg.element_spacing,
                wavelength=cfg.wavelength,
            )

            h_los = math.sqrt(beta_ru) * a_ru
            h_nlos = complex_gaussian((cfg.N,), scale=math.sqrt(beta_ru), rng=self.rng)

            zeta = cfg.zeta_ru
            h = math.sqrt(zeta / (zeta + 1.0)) * h_los + math.sqrt(1.0 / (zeta + 1.0)) * h_nlos
            out[k] = h

        return out

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """一次性返回 BS-RIS 信道和 RIS-用户信道样本。"""
        h_br = self.generate_bs_ris()
        h_ru = self.generate_ris_users()
        return h_br, h_ru
