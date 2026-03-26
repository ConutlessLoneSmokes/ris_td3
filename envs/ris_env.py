# envs/ris_env.py
import numpy as np
from configs.default import SystemConfig
from envs.channel_model import ChannelGenerator
from envs.constraints import map_raw_theta, map_raw_cbl, map_raw_beamforming
from envs.fbl import sinr_all, reward_total_fbl, cascaded_channel, ris_coefficients

class RISEnv:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.channel_gen = ChannelGenerator(cfg)
        
        # 定义动作维度：CBL变量(K) + 波束赋形幅度(K*M) + 波束赋形相位(K*M) + 智能表面相移(N)
        self.action_dim = cfg.K + 2 * cfg.K * cfg.M + cfg.N
        self.state_dim = 0  # 状态维度将在首次调用_get_state时动态确定
        
        self.reset()
        
    def reset(self):
        """初始化网络拓扑并重置智能体的观测历史"""
        self.h_br, self.h_ru = self.channel_gen.sample()
        
        # 赋予历史动作和奖励初始值
        self.last_theta = np.zeros(self.cfg.N)
        # 采用均匀分配的功率初始化波束赋形矩阵
        self.last_w = np.ones((self.cfg.K, self.cfg.M), dtype=np.complex128) * \
                      np.sqrt(self.cfg.p_total_watt / (self.cfg.K * self.cfg.M))
        self.last_reward = 0.0
        self.current_step = 0
        return self._get_state()
        
    def _get_state(self):
        """遵循论文公式(27)-(29)提取一维状态特征张量"""
        cfg = self.cfg
        theta_vec = ris_coefficients(self.last_theta, cfg)
        
        s1, s2 = [], []
        
        for k in range(cfg.K):
            # 计算复合信道矩阵及其等效向量
            h_tilde_k = cascaded_channel(self.h_ru[k], self.h_br)  # 维度: N x M
            upsilon_k = np.conjugate(theta_vec) @ h_tilde_k        # 维度: 1 x M
            
            # 填充 s_t^2: 范数与相位特征
            s2.extend([np.linalg.norm(upsilon_k), np.linalg.norm(self.last_w[k])])
            s2.extend(np.angle(upsilon_k).flatten())
            s2.extend(np.angle(h_tilde_k).flatten())
            s2.extend(np.angle(self.last_w[k]).flatten())
            
            # 填充 s_t^1: 目标信号与干扰项
            for kp in range(cfg.K):
                upsilon_kkp = upsilon_k @ self.last_w[kp]
                s1.extend([np.abs(upsilon_kkp), np.angle(upsilon_kkp)])
                
        # 填充 s_t^3 (历史相移) 与 s_t^4 (历史奖励)
        s3 = self.last_theta.tolist()
        s4 = [self.last_reward]
        
        state = np.array(s1 + s2 + s3 + s4, dtype=np.float32)
        self.state_dim = len(state)
        return state

    def step(self, action: np.ndarray):
        """接收策略网络的输出参数，映射至物理环境并执行交互"""
        cfg = self.cfg
        idx = 0
        
        # 解包连续动作向量
        raw_cbl = action[idx : idx + cfg.K]
        idx += cfg.K
        
        raw_mag = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M
        
        raw_phase = action[idx : idx + cfg.K * cfg.M].reshape(cfg.K, cfg.M)
        idx += cfg.K * cfg.M
        
        raw_theta = action[idx : idx + cfg.N]
        
        # 应用物理约束进行参数映射
        cbl = map_raw_cbl(raw_cbl, cfg)
        w = map_raw_beamforming(raw_mag, raw_phase, cfg)
        theta = map_raw_theta(raw_theta)
        
        # 获取环境反馈
        sinr = sinr_all(self.h_br, self.h_ru, theta, w, cfg)
        reward = reward_total_fbl(sinr, cbl, cfg.target_error_prob)
        
        # 迭代历史观测数据
        self.last_theta = theta
        self.last_w = w
        self.last_reward = reward
        
        next_state = self._get_state()
        self.current_step += 1

        # 论文设定场景为单次传输(Single-shot transmission)，在此处将done设定为True标识回合结束
        done = bool(self.current_step >= getattr(self.cfg, 'max_steps', 100)) 
        
        info = {
            "sinr": sinr.astype(np.float64, copy=True),
            "cbl": cbl.astype(np.float64, copy=True),
            "theta": theta.astype(np.float64, copy=True),
            "power": float(np.sum(np.abs(w) ** 2)),
        }

        return next_state, reward, done, info
