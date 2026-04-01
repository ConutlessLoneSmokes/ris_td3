from __future__ import annotations

import numpy as np

from core.types import Metrics
from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.constraints import compute_constraint_violations
from problems.ris_miso_urllc.objective import reward_total_fbl, ris_coefficients, sinr_all
from problems.ris_miso_urllc.types import ProblemInstance, Solution


class Evaluator:
    """问题层统一评估器。"""

    def __init__(self, cfg: ProblemConfig):
        self.cfg = cfg

    def evaluate(self, instance: ProblemInstance, solution: Solution) -> Metrics:
        """对给定问题实例与候选解进行统一评估。"""
        sinr = sinr_all(
            h_br=instance.h_br,
            h_ru=instance.h_ru,
            theta=solution.theta,
            beamforming=solution.beamforming,
            cfg=self.cfg,
        )
        reward = reward_total_fbl(sinr, solution.cbl, self.cfg.target_error_prob)
        power = float(np.sum(np.abs(solution.beamforming) ** 2))
        violations = compute_constraint_violations(solution, self.cfg)

        theta_coeff = ris_coefficients(solution.theta, self.cfg)
        extra = {
            "mean_reflection_amplitude": float(np.mean(np.abs(theta_coeff))),
        }
        return Metrics(
            reward=reward,
            sinr=sinr.astype(np.float64),
            cbl=np.asarray(solution.cbl, dtype=np.float64),
            power=power,
            constraint_violations=violations,
            extra=extra,
        )
