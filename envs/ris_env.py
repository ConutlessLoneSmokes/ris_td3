import numpy as np

from configs.default import SystemConfig
from envs.channel_model import ChannelGenerator
from envs.constraints import map_raw_beamforming, map_raw_cbl, map_raw_theta
from envs.fbl import cascaded_channel, reward_total_fbl, ris_coefficients, sinr_all


class RISEnv:
    """RIS 辅助 MISO-URLLC 环境，用于与 TD3 智能体交互。"""

    def __init__(self, cfg: SystemConfig):
        """保存配置、构造信道生成器并初始化环境维度。"""
        self.cfg = cfg
        self.channel_gen = ChannelGenerator(cfg)

        # 动作由三部分组成：CBL、波束赋形幅度/相位以及 RIS 相位。
        self.action_dim = cfg.K + 2 * cfg.K * cfg.M + cfg.N
        self.state_dim = 0  # 状态维度会在首次生成状态后动态得到。

        self.reset()

    def reset(self):
        """重置环境，重新采样信道并初始化历史动作。"""
        self.h_br, self.h_ru = self.channel_gen.sample()

        # 初始化历史 RIS 相位、波束赋形和奖励，供状态构造函数使用。
        self.last_theta = np.zeros(self.cfg.N)
        self.last_w = np.ones((self.cfg.K, self.cfg.M), dtype=np.complex128) * np.sqrt(
            self.cfg.p_total_watt / (self.cfg.K * self.cfg.M)
        )
        self.last_reward = 0.0
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """根据论文中的状态定义构造一维观测向量。"""
        cfg = self.cfg
        theta_vec = ris_coefficients(self.last_theta, cfg)

        s1, s2 = [], []

        for k in range(cfg.K):
            # 第 k 个用户的级联信道和等效观测向量。
            h_tilde_k = cascaded_channel(self.h_ru[k], self.h_br)
            upsilon_k = np.conjugate(theta_vec) @ h_tilde_k

            # s2 收集范数和相位等局部特征。
            s2.extend([np.linalg.norm(upsilon_k), np.linalg.norm(self.last_w[k])])
            s2.extend(np.angle(upsilon_k).flatten())
            s2.extend(np.angle(h_tilde_k).flatten())
            s2.extend(np.angle(self.last_w[k]).flatten())

            # s1 收集目标信号与多用户干扰对应的复数幅度和相位。
            for kp in range(cfg.K):
                upsilon_kkp = upsilon_k @ self.last_w[kp]
                s1.extend([np.abs(upsilon_kkp), np.angle(upsilon_kkp)])

        # s3 为历史 RIS 相位，s4 为上一时刻奖励。
        s3 = self.last_theta.tolist()
        s4 = [self.last_reward]

        state = np.array(s1 + s2 + s3 + s4, dtype=np.float32)
        self.state_dim = len(state)
        return state

    def step(self, action: np.ndarray):
        """执行一步交互：动作解包、物理映射、奖励计算和状态更新。"""
        cfg = self.cfg
        idx = 0

        # 依次从连续动作向量中切分出各个物理量。
        raw_cbl = action[idx : idx + cfg.K]
        idx += cfg.K

        raw_mag = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M

        raw_phase = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M

        raw_theta = action[idx : idx + cfg.N]

        # 将网络原始输出映射到满足约束的物理变量。
        cbl = map_raw_cbl(raw_cbl, cfg)
        w = map_raw_beamforming(raw_mag, raw_phase, cfg)
        theta = map_raw_theta(raw_theta)

        # 计算当前动作对应的链路质量和有限块长奖励。
        sinr = sinr_all(self.h_br, self.h_ru, theta, w, cfg)
        reward = reward_total_fbl(sinr, cbl, cfg.target_error_prob)

        # 将本步结果缓存下来，以便下一次构造状态时使用。
        self.last_theta = theta
        self.last_w = w
        self.last_reward = reward

        next_state = self._get_state()
        self.current_step += 1

        # 当前代码允许按 max_steps 控制 episode 长度，便于训练与调试。
        done = bool(self.current_step >= getattr(self.cfg, "max_steps", 100))

        info = {
            "sinr": sinr.astype(np.float64, copy=True),
            "cbl": cbl.astype(np.float64, copy=True),
            "theta": theta.astype(np.float64, copy=True),
            "power": float(np.sum(np.abs(w) ** 2)),
        }

        return next_state, reward, done, info
