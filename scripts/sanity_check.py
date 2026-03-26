# scripts/sanity_check.py
import numpy as np

from configs.default import SystemConfig
from envs.channel_model import ChannelGenerator
from envs.constraints import map_raw_theta, map_raw_cbl, map_raw_beamforming
from envs.fbl import sinr_all, reward_total_fbl, ris_coefficients


def main():
    cfg = SystemConfig()
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

    print("H_br shape:", h_br.shape)
    print("H_ru shape:", h_ru.shape)
    print("theta shape:", theta.shape)
    print("w shape:", w.shape)
    print("cbl:", cbl)
    print("sum cbl:", cbl.sum())
    print("min cbl:", cbl.min())
    print("total power:", np.sum(np.abs(w) ** 2))
    print("RIS amplitude range:", np.min(np.abs(coeff)), np.max(np.abs(coeff)))
    print("SINR:", sinr)
    print("Reward:", reward)


if __name__ == "__main__":
    main()