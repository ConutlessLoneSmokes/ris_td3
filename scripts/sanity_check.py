from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接从项目根目录调用脚本。
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from agents.replay_buffer import ReplayBuffer
from agents.td3 import TD3Agent
from configs.default import SystemConfig
from envs.channel_model import ChannelGenerator
from envs.constraints import map_raw_beamforming, map_raw_cbl, map_raw_theta
from envs.fbl import reward_total_fbl, ris_coefficients, sinr_all
from envs.ris_env import RISEnv


def main() -> None:
    """执行环境、网络、缓存与 checkpoint 的最小闭环冒烟测试。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = replace(
            SystemConfig(),
            actor_hidden_dims=(64, 32, 16),
            critic_state_hidden_dim=64,
            critic_action_hidden_dim=64,
            critic_hidden_dims=(48, 32),
            batch_size=4,
            buffer_size=32,
            warmup_episodes=0,
            log_dir=str(Path(tmpdir) / "tb"),
            ckpt_dir=str(Path(tmpdir) / "ckpt"),
        )

        gen = ChannelGenerator(cfg)
        h_br, h_ru = gen.sample()

        rng = np.random.default_rng(cfg.seed + 1)
        raw_theta = rng.uniform(-1.0, 1.0, size=(cfg.N,))
        raw_cbl = rng.uniform(-1.0, 1.0, size=(cfg.K,))
        raw_mag = rng.uniform(-1.0, 1.0, size=(cfg.K, cfg.M))
        raw_phase = rng.uniform(-1.0, 1.0, size=(cfg.K, cfg.M))

        theta = map_raw_theta(raw_theta)
        cbl = map_raw_cbl(raw_cbl, cfg)
        w = map_raw_beamforming(raw_mag, raw_phase, cfg)

        sinr = sinr_all(h_br, h_ru, theta, w, cfg)
        reward = reward_total_fbl(sinr, cbl, cfg.target_error_prob)
        coeff = ris_coefficients(theta, cfg)

        # 基础数学与形状检查。
        assert h_br.shape == (cfg.N, cfg.M)
        assert h_ru.shape == (cfg.K, cfg.N)
        assert theta.shape == (cfg.N,)
        assert w.shape == (cfg.K, cfg.M)
        assert np.isclose(np.sum(cbl), cfg.total_cbl, atol=1e-6)
        assert cbl.min() >= cfg.min_cbl - 1e-6
        assert np.isfinite(sinr).all()
        assert np.isfinite(reward)
        assert np.all(np.abs(coeff) >= cfg.beta_min - 1e-6)

        env = RISEnv(cfg)
        state = env.reset()
        assert env.state_dim > 0
        assert state.shape == (env.state_dim,)

        agent = TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg)
        actor_action = agent.select_action(state, deterministic=True)
        assert actor_action.shape == (env.action_dim,)
        assert np.all(actor_action <= 1.0 + 1e-6)
        assert np.all(actor_action >= -1.0 - 1e-6)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        action_tensor = torch.as_tensor(actor_action, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q1, q2 = agent.critic(state_tensor, action_tensor)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)

        buffer = ReplayBuffer(state_dim=env.state_dim, action_dim=env.action_dim, capacity=cfg.buffer_size)
        for _ in range(cfg.batch_size):
            episode_state = env.reset()
            random_action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            next_state, episode_reward, done, info = env.step(random_action)
            buffer.add(episode_state, random_action, episode_reward, next_state, done)
            assert np.isfinite(info["sinr"]).all()
            assert np.isfinite(info["power"])

        sampled = buffer.sample(cfg.batch_size, agent.device)
        assert sampled[0].shape == (cfg.batch_size, env.state_dim)
        assert sampled[1].shape == (cfg.batch_size, env.action_dim)

        losses = agent.train_step(buffer)
        assert losses is not None
        assert np.isfinite(losses["critic_loss"])
        assert np.isfinite(losses["actor_loss"])

        checkpoint_path = Path(tmpdir) / "td3_smoke.pt"
        agent.save(checkpoint_path)
        restored_agent = TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg)
        restored_agent.load(checkpoint_path)

        reference_action = agent.select_action(state, deterministic=True)
        restored_action = restored_agent.select_action(state, deterministic=True)
        assert np.allclose(reference_action, restored_action, atol=1e-6)

        print("环境数学检查通过。")
        print("状态维度:", env.state_dim)
        print("动作维度:", env.action_dim)
        print("SINR:", sinr)
        print("Reward:", reward)
        print("单步训练检查通过，critic_loss =", losses["critic_loss"])
        print("checkpoint 保存/加载检查通过。")


if __name__ == "__main__":
    main()
