from __future__ import annotations

import numpy as np

from core.types import Metrics
from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.encoding import ContinuousActionCodec, ObservationEncoder, build_default_solution
from problems.ris_miso_urllc.evaluator import Evaluator
from problems.ris_miso_urllc.scenario import ScenarioSampler
from problems.ris_miso_urllc.types import ProblemInstance, Solution


class RISEnv:
    """仅服务 RL 方法的环境适配器。"""

    def __init__(
        self,
        cfg: ProblemConfig,
        sampler: ScenarioSampler | None = None,
        evaluator: Evaluator | None = None,
        action_codec: ContinuousActionCodec | None = None,
        observation_encoder: ObservationEncoder | None = None,
    ):
        self.cfg = cfg
        self.sampler = sampler or ScenarioSampler(cfg)
        self.evaluator = evaluator or Evaluator(cfg)
        self.action_codec = action_codec or ContinuousActionCodec(cfg)
        self.observation_encoder = observation_encoder or ObservationEncoder(cfg)

        self.action_dim = self.action_codec.action_dim
        self.state_dim = 0
        self.current_step = 0
        self.instance: ProblemInstance | None = None
        self.last_solution: Solution | None = None
        self.last_metrics: Metrics | None = None
        self.last_reward = 0.0

        self.reset()

    def reset(self, instance: ProblemInstance | None = None) -> np.ndarray:
        """重置到新的信道场景或用户给定场景。"""
        self.instance = instance if instance is not None else self.sampler.sample()
        self.last_solution = build_default_solution(self.cfg)
        self.last_reward = 0.0
        self.last_metrics = None
        self.current_step = 0
        state = self.observation_encoder.encode(self.instance, self.last_solution, self.last_reward)
        self.state_dim = int(state.shape[0])
        return state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """执行一步交互。"""
        assert self.instance is not None

        solution = self.action_codec.decode(action)
        metrics = self.evaluator.evaluate(self.instance, solution)

        self.last_solution = solution
        self.last_metrics = metrics
        self.last_reward = float(metrics.reward)
        self.current_step += 1

        next_state = self.observation_encoder.encode(self.instance, solution, self.last_reward)
        done = bool(self.current_step >= self.cfg.max_steps)

        info = {
            "sinr": metrics.sinr.astype(np.float64, copy=True),
            "cbl": metrics.cbl.astype(np.float64, copy=True),
            "theta": solution.theta.astype(np.float64, copy=True),
            "power": float(metrics.power),
            "constraint_violations": dict(metrics.constraint_violations),
            "solution": solution,
            "metrics": metrics,
        }
        return next_state, float(metrics.reward), done, info
