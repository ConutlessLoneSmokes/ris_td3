from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from configs.default import build_problem_config
from core.io import build_experiment_paths
from problems.ris_miso_urllc.constraints import compute_constraint_violations, map_raw_beamforming, map_raw_cbl, map_raw_theta
from problems.ris_miso_urllc.evaluator import Evaluator
from problems.ris_miso_urllc.objective import reward_total_fbl, ris_coefficients, sinr_all
from problems.ris_miso_urllc.rl_env import RISEnv
from problems.ris_miso_urllc.scenario import ScenarioSampler
from runners.common import evaluate_solver_over_instances
from solvers.baselines.random_search import RandomSearchConfig, RandomSearchSolver
from solvers.rl.ddpg import DDPGConfig, DDPGSolver
from solvers.rl.replay_buffer import ReplayBuffer
from solvers.rl.td3 import TD3Config, TD3Solver


def main() -> None:
    """执行新架构下的最小闭环冒烟测试。"""
    run_root = Path("d:/PyTest/ris_td3/outputs/sanity_tmp")
    run_root.mkdir(parents=True, exist_ok=True)

    problem_cfg = replace(
        build_problem_config(),
        seed=123,
        train_episodes=8,
        eval_episodes=3,
        max_steps=4,
        outputs_root=str(run_root),
    )
    td3_cfg = replace(
        TD3Config(),
        actor_hidden_dims=(64, 32, 16),
        critic_state_hidden_dim=64,
        critic_action_hidden_dim=64,
        critic_hidden_dims=(48, 32),
        batch_size=4,
        buffer_size=32,
        warmup_episodes=0,
        device="cpu",
    )
    ddpg_cfg = replace(
        DDPGConfig(),
        actor_hidden_dims=(64, 32, 16),
        critic_state_hidden_dim=64,
        critic_action_hidden_dim=64,
        critic_hidden_dims=(48, 32),
        batch_size=4,
        buffer_size=32,
        warmup_episodes=0,
        device="cpu",
    )

    sampler_a = ScenarioSampler(problem_cfg, seed=problem_cfg.seed)
    sampler_b = ScenarioSampler(problem_cfg, seed=problem_cfg.seed)
    instance_a = sampler_a.sample()
    instance_b = sampler_b.sample()
    assert np.allclose(instance_a.h_br, instance_b.h_br)
    assert np.allclose(instance_a.h_ru, instance_b.h_ru)

    rng = np.random.default_rng(problem_cfg.seed + 1)
    raw_theta = rng.uniform(-1.0, 1.0, size=(problem_cfg.N,))
    raw_cbl = rng.uniform(-1.0, 1.0, size=(problem_cfg.K,))
    raw_mag = rng.uniform(-1.0, 1.0, size=(problem_cfg.K, problem_cfg.M))
    raw_phase = rng.uniform(-1.0, 1.0, size=(problem_cfg.K, problem_cfg.M))

    theta = map_raw_theta(raw_theta)
    cbl = map_raw_cbl(raw_cbl, problem_cfg)
    beamforming = map_raw_beamforming(raw_mag, raw_phase, problem_cfg)

    sinr = sinr_all(instance_a.h_br, instance_a.h_ru, theta, beamforming, problem_cfg)
    reward = reward_total_fbl(sinr, cbl, problem_cfg.target_error_prob)
    coeff = ris_coefficients(theta, problem_cfg)
    violations = compute_constraint_violations(type("Tmp", (), {"theta": theta, "beamforming": beamforming, "cbl": cbl})(), problem_cfg)

    assert theta.shape == (problem_cfg.N,)
    assert beamforming.shape == (problem_cfg.K, problem_cfg.M)
    assert np.isclose(np.sum(cbl), problem_cfg.total_cbl, atol=1e-6)
    assert np.isfinite(sinr).all()
    assert np.isfinite(reward)
    assert np.all(np.abs(coeff) >= problem_cfg.beta_min - 1e-6)
    assert all(value >= 0.0 for value in violations.values())

    env = RISEnv(problem_cfg)
    state = env.reset()
    assert env.state_dim > 0
    assert state.shape == (env.state_dim,)

    td3_solver = TD3Solver(problem_cfg, td3_cfg)
    td3_solver.bind_environment(env.state_dim, env.action_dim)
    action = td3_solver.select_action(state, deterministic=True)
    assert action.shape == (env.action_dim,)
    buffer = ReplayBuffer(env.state_dim, env.action_dim, td3_cfg.buffer_size)
    for _ in range(td3_cfg.batch_size):
        sample_state = env.reset()
        random_action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
        next_state, sample_reward, done, _ = env.step(random_action)
        buffer.add(sample_state, random_action, sample_reward * td3_cfg.reward_scale, next_state, done)
    td3_losses = td3_solver.update(buffer)
    assert td3_losses is not None
    assert np.isfinite(td3_losses["critic_loss"])

    ddpg_solver = DDPGSolver(problem_cfg, ddpg_cfg)
    ddpg_solver.bind_environment(env.state_dim, env.action_dim)
    ddpg_losses = ddpg_solver.update(buffer)
    assert ddpg_losses is not None
    assert np.isfinite(ddpg_losses["critic_loss"])
    assert np.isfinite(ddpg_losses["actor_loss"])

    random_solver = RandomSearchSolver(problem_cfg, RandomSearchConfig(num_candidates=16))
    evaluator = Evaluator(problem_cfg)
    random_solution = random_solver.solve(instance_a)
    random_metrics = evaluator.evaluate(instance_a, random_solution)
    assert np.isfinite(random_metrics.reward)

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    td3_path = ckpt_dir / "td3.pt"
    td3_solver.save(td3_path)
    restored_td3 = TD3Solver(problem_cfg, td3_cfg)
    restored_td3.load(td3_path)
    restored_action = restored_td3.select_action(state, deterministic=True)
    assert np.allclose(action, restored_action, atol=1e-6)

    eval_summary = evaluate_solver_over_instances("ris_miso_urllc", problem_cfg, random_solver, 2)
    assert "avg_reward" in eval_summary

    paths = build_experiment_paths(problem_cfg.outputs_root, problem_cfg.name, "random_search", "smoke")
    assert paths.run_dir.exists()

    print("问题层与评估器检查通过。")
    print("状态维度:", env.state_dim)
    print("动作维度:", env.action_dim)
    print("TD3 单步更新检查通过，critic_loss =", td3_losses["critic_loss"])
    print("DDPG 单步更新检查通过，critic_loss =", ddpg_losses["critic_loss"])
    print("随机搜索基线检查通过，avg_reward =", eval_summary["avg_reward"])
    print("checkpoint 保存/加载检查通过。")


if __name__ == "__main__":
    main()
