from __future__ import annotations

import numpy as np

from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.constraints import map_raw_beamforming, map_raw_cbl, map_raw_theta
from problems.ris_miso_urllc.objective import cascaded_channel, ris_coefficients
from problems.ris_miso_urllc.types import ProblemInstance, Solution


def build_default_solution(cfg: ProblemConfig) -> Solution:
    """构造环境重置后使用的默认初始解。"""
    theta = np.zeros(cfg.N, dtype=np.float64)
    beamforming = np.ones((cfg.K, cfg.M), dtype=np.complex128) * np.sqrt(cfg.p_total_watt / (cfg.K * cfg.M))
    cbl = np.full(cfg.K, cfg.total_cbl / cfg.K, dtype=np.float64)
    return Solution(theta=theta, beamforming=beamforming, cbl=cbl)


class ContinuousActionCodec:
    """将连续动作向量与物理可行解之间做双向语义绑定。"""

    def __init__(self, cfg: ProblemConfig):
        self.cfg = cfg
        self.action_dim = cfg.K + 2 * cfg.K * cfg.M + cfg.N

    def decode(self, action: np.ndarray) -> Solution:
        """将原始连续动作解码成物理层候选解。"""
        cfg = self.cfg
        idx = 0

        raw_cbl = action[idx : idx + cfg.K]
        idx += cfg.K

        raw_mag = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M

        raw_phase = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M

        raw_theta = action[idx : idx + cfg.N]

        return Solution(
            theta=map_raw_theta(raw_theta),
            beamforming=map_raw_beamforming(raw_mag, raw_phase, cfg),
            cbl=map_raw_cbl(raw_cbl, cfg),
        )

    def sample_random_action(self, rng: np.random.Generator) -> np.ndarray:
        """在归一化动作空间中采样随机动作。"""
        return rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)


class ObservationEncoder:
    """将问题实例与历史信息编码为 RL 可用状态向量。"""

    def __init__(self, cfg: ProblemConfig):
        self.cfg = cfg

    def encode(
        self,
        instance: ProblemInstance,
        last_solution: Solution,
        last_reward: float,
    ) -> np.ndarray:
        """按当前 TD3 版本的状态定义构造一维观测。"""
        cfg = self.cfg
        theta_vec = ris_coefficients(last_solution.theta, cfg)

        s1: list[float] = []
        s2: list[float] = []

        for k in range(cfg.K):
            h_tilde_k = cascaded_channel(instance.h_ru[k], instance.h_br)
            upsilon_k = np.conjugate(theta_vec) @ h_tilde_k

            s2.extend([float(np.linalg.norm(upsilon_k)), float(np.linalg.norm(last_solution.beamforming[k]))])
            s2.extend(np.angle(upsilon_k).flatten().tolist())
            s2.extend(np.angle(h_tilde_k).flatten().tolist())
            s2.extend(np.angle(last_solution.beamforming[k]).flatten().tolist())

            for kp in range(cfg.K):
                upsilon_kkp = upsilon_k @ last_solution.beamforming[kp]
                s1.extend([float(np.abs(upsilon_kkp)), float(np.angle(upsilon_kkp))])

        s3 = last_solution.theta.astype(np.float64).tolist()
        s4 = [float(last_reward)]
        return np.array(s1 + s2 + s3 + s4, dtype=np.float32)
