from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接从项目根目录调用脚本。
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agents.td3 import TD3Agent
from configs.default import SystemConfig
from envs.ris_env import RISEnv


def resolve_latest_run(outputs_root: str) -> Path | None:
    """读取最新一次训练对应的实验目录。"""
    latest_run_file = Path(outputs_root) / "latest_run.txt"
    if not latest_run_file.exists():
        return None
    return Path(latest_run_file.read_text(encoding="utf-8").strip())


def parse_args() -> argparse.Namespace:
    """解析评估脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="RIS-TD3 评估脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="待加载的 checkpoint 路径；若不提供，则优先读取最新实验目录中的 best.pt",
    )
    parser.add_argument("--eval-episodes", type=int, default=None, help="覆盖默认评估回合数")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SystemConfig:
    """根据命令行参数构造评估配置。"""
    cfg = SystemConfig()
    if args.eval_episodes is not None:
        cfg = replace(cfg, eval_episodes=args.eval_episodes)
    return cfg


def main() -> None:
    """加载训练好的模型并输出评估统计结果。"""
    args = parse_args()
    cfg = build_config(args)

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        latest_run = resolve_latest_run(cfg.outputs_root)
        if latest_run is not None:
            checkpoint_path = latest_run / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = latest_run / "checkpoints" / "latest.pt"
        else:
            checkpoint_path = Path(cfg.ckpt_dir) / "latest.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    env = RISEnv(cfg)
    initial_state = env.reset()
    agent = TD3Agent(state_dim=initial_state.shape[0], action_dim=env.action_dim, cfg=cfg)
    agent.load(checkpoint_path)

    rewards = []
    mean_sinrs = []
    sum_cbls = []
    total_powers = []

    for _ in range(cfg.eval_episodes):
        state = env.reset()
        action = agent.select_action(state, deterministic=True)
        _, reward, _, info = env.step(action)

        rewards.append(float(reward))
        mean_sinrs.append(float(np.mean(info["sinr"])))
        sum_cbls.append(float(np.sum(info["cbl"])))
        total_powers.append(float(info["power"]))

    print(f"checkpoint: {checkpoint_path}")
    print(f"episodes: {cfg.eval_episodes}")
    print(f"avg_reward: {np.mean(rewards):.6f}")
    print(f"avg_mean_sinr: {np.mean(mean_sinrs):.6f}")
    print(f"avg_sum_cbl: {np.mean(sum_cbls):.6f}")
    print(f"avg_total_power: {np.mean(total_powers):.6f}")

    checkpoint_dir = checkpoint_path.parent
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    evaluation_summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "episodes": cfg.eval_episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_mean_sinr": float(np.mean(mean_sinrs)),
        "avg_sum_cbl": float(np.mean(sum_cbls)),
        "avg_total_power": float(np.mean(total_powers)),
    }
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
