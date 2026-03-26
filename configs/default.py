# configs/default.py
from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class SystemConfig:
    # topology
    K: int = 4
    Mx: int = 2
    My: int = 2
    Nx: int = 4
    Ny: int = 4

    # radio parameters
    bandwidth_hz: float = 1e5
    noise_density_dbm_hz: float = -174.0
    noise_figure_db: float = 3.0
    p_total_watt: float = 1e-3
    wavelength: float = 0.1
    element_spacing: float = 0.05

    # finite blocklength
    total_cbl: float = 100.0
    min_cbl: float = 10.0
    target_error_prob: float = 1e-8

    # rician factors
    zeta_br: float = 10.0
    zeta_ru: float = 10.0

    # pathloss
    pl0_db: float = -30.0
    pathloss_exp: float = 2.2

    # geometry
    bs_pos: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ris_pos: List[float] = field(default_factory=lambda: [40.0, 0.0])
    user_pos: List[List[float]] = field(
        default_factory=lambda: [
            [16.0, 40.0],
            [32.0, 40.0],
            [48.0, 40.0],
            [64.0, 40.0],
        ]
    )

    # heights
    bs_height: float = 12.5
    ris_height: float = 12.5
    user_height: float = 1.5

    # practical RIS
    beta_min: float = 0.4
    alpha_ris: float = 1.9
    phi_ris: float = 0.43 * math.pi

    # environment
    seed: int = 42
    eps_div: float = 1e-12

    @property
    def M(self) -> int:
        return self.Mx * self.My

    @property
    def N(self) -> int:
        return self.Nx * self.Ny

    @property
    def sigma2(self) -> float:
        noise_dbm = self.noise_density_dbm_hz + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 1e-3 * 10 ** (noise_dbm / 10.0)