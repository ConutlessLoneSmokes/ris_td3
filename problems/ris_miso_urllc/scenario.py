from __future__ import annotations

import numpy as np

from problems.ris_miso_urllc.channel import ChannelGenerator
from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.types import ProblemInstance


class ScenarioSampler:
    """统一负责问题实例采样。"""

    def __init__(self, cfg: ProblemConfig, seed: int | None = None):
        self.cfg = cfg
        self.channel_generator = ChannelGenerator(cfg, seed=seed)

    def sample(self) -> ProblemInstance:
        """采样一次固定信道 realization。"""
        h_br = self.channel_generator.generate_bs_ris()
        h_ru = self.channel_generator.generate_ris_users()
        bs_xyz = self.channel_generator.bs_xyz()
        ris_xyz = self.channel_generator.ris_xyz()
        user_xyz = np.stack([self.channel_generator.user_xyz(k) for k in range(self.cfg.K)], axis=0)
        return ProblemInstance(
            h_br=h_br,
            h_ru=h_ru,
            bs_xyz=bs_xyz,
            ris_xyz=ris_xyz,
            user_xyz=user_xyz,
            sigma2=self.cfg.sigma2,
            p_total_watt=self.cfg.p_total_watt,
        )
