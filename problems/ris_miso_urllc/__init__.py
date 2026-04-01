"""RIS-aided MISO URLLC 问题定义。"""

from core.registry import PROBLEM_REGISTRY, register_problem
from problems.ris_miso_urllc.config import ProblemConfig
from problems.ris_miso_urllc.encoding import ContinuousActionCodec, ObservationEncoder
from problems.ris_miso_urllc.evaluator import Evaluator
from problems.ris_miso_urllc.rl_env import RISEnv
from problems.ris_miso_urllc.scenario import ScenarioSampler


def build_problem_components(cfg: ProblemConfig) -> dict[str, object]:
    """构造问题相关组件。"""
    return {
        "config": cfg,
        "sampler": ScenarioSampler(cfg),
        "evaluator": Evaluator(cfg),
        "action_codec": ContinuousActionCodec(cfg),
        "observation_encoder": ObservationEncoder(cfg),
        "env_cls": RISEnv,
    }


def ensure_registered() -> None:
    """将该问题注册到全局注册器。"""
    if "ris_miso_urllc" in PROBLEM_REGISTRY:
        return
    register_problem(
        name="ris_miso_urllc",
        config_factory=ProblemConfig,
        builder=build_problem_components,
    )


__all__ = [
    "ProblemConfig",
    "ScenarioSampler",
    "Evaluator",
    "ContinuousActionCodec",
    "ObservationEncoder",
    "RISEnv",
    "ensure_registered",
]

