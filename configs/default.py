from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class SystemConfig:
    """项目统一配置，集中管理系统参数、训练超参数和输出路径。"""

    # 系统拓扑参数
    K: int = 4  # 用户/执行器数量
    Mx: int = 2  # 基站 UPA 天线在 x 方向的排布数
    My: int = 2  # 基站 UPA 天线在 y 方向的排布数
    Nx: int = 4  # RIS 单元在 x 方向的排布数
    Ny: int = 4  # RIS 单元在 y 方向的排布数

    # 无线物理层参数
    bandwidth_hz: float = 1e5  # 系统带宽，单位 Hz
    noise_density_dbm_hz: float = -174.0  # 噪声功率谱密度，单位 dBm/Hz
    noise_figure_db: float = 3.0  # 接收机噪声系数，单位 dB
    p_total_watt: float = 1e-3  # 基站总发射功率约束，单位 W
    wavelength: float = 0.1  # 载波波长，单位 m
    element_spacing: float = 0.05  # UPA 相邻天线/单元间距，单位 m

    # 有限块长通信参数
    total_cbl: float = 100.0  # 系统总可分配信道块长
    min_cbl: float = 10.0  # 单用户最小信道块长
    target_error_prob: float = 1e-12  # 目标误块率/误包率

    # Rician 信道参数
    zeta_br: float = 10.0  # BS-RIS 链路的 Rician 因子
    zeta_ru: float = 10.0  # RIS-用户链路的 Rician 因子

    # 路损模型参数
    pl0_db: float = -30.0  # 参考路损常数，单位 dB
    pathloss_exp: float = 2.2  # 路损指数

    # 二维平面几何位置
    bs_pos: List[float] = field(default_factory=lambda: [0.0, 0.0])  # 基站在平面内的位置
    ris_pos: List[float] = field(default_factory=lambda: [40.0, 0.0])  # RIS 在平面内的位置
    user_pos: List[List[float]] = field(
        default_factory=lambda: [
            [16.0, 40.0],
            [32.0, 40.0],
            [48.0, 40.0],
            [64.0, 40.0],
        ]
    )  # 多个用户在平面内的位置

    # 三维高度参数
    bs_height: float = 12.5  # 基站高度，单位 m
    ris_height: float = 12.5  # RIS 高度，单位 m
    user_height: float = 1.5  # 用户高度，单位 m

    # 实际 RIS 幅相耦合参数
    beta_min: float = 0.4  # 最小反射幅度
    alpha_ris: float = 1.9  # 幅相曲线陡峭程度
    phi_ris: float = 0.43 * math.pi  # 幅相曲线水平偏移量

    # 通用数值与随机性参数
    seed: int = 42  # 全局随机种子
    eps_div: float = 1e-30  # 防止除零的小常数

    # Actor 网络结构参数
    actor_hidden_dims: Tuple[int, int, int] = (800, 400, 200)  # Actor 各隐藏层宽度

    # Critic 网络结构参数
    critic_state_hidden_dim: int = 800  # Critic 中状态分支第一层宽度
    critic_action_hidden_dim: int = 800  # Critic 中动作分支第一层宽度
    critic_hidden_dims: Tuple[int, int] = (600, 400)  # Critic 融合后的隐藏层宽度
    use_layer_norm: bool = True  # 是否在网络内部使用 LayerNorm

    # TD3 优化超参数
    actor_lr: float = 1e-4  # Actor 学习率
    critic_lr: float = 1e-4  # Critic 学习率
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005  # 目标网络软更新系数
    policy_noise: float = math.sqrt(0.1)  # 目标动作平滑噪声标准差
    noise_clip: float = 0.5  # 目标动作噪声裁剪阈值
    exploration_noise: float = math.sqrt(0.1)  # 训练时动作探索噪声标准差
    policy_delay: int = 4  # Actor 相对 Critic 的延迟更新步数

    # 训练过程参数
    buffer_size: int = 10000  # 经验回放缓存容量
    batch_size: int = 64  # 每次参数更新的采样批大小
    warmup_episodes: int = 0  # 纯随机探索的预热回合数
    train_episodes: int = 10000  # 总训练回合数
    eval_interval: int = 500  # 训练过程中执行评估的间隔回合数
    save_interval: int = 1000  # 训练过程中保存 checkpoint 的间隔回合数
    eval_episodes: int = 100  # 每次评估使用的独立信道实现数
    max_steps: int = 100  # 每个 episode 的最大交互步数
    device: str = "cuda"  # 训练设备优先级，若不可用会在代码中自动回退到 CPU

    # 输出目录参数
    outputs_root: str = "outputs/experiments"  # 全部实验输出的根目录
    log_dir: str = "outputs/experiments/default/tb"  # TensorBoard 默认输出目录
    ckpt_dir: str = "outputs/experiments/default/checkpoints"  # checkpoint 默认输出目录

    @property
    def M(self) -> int:
        """返回基站总天线数。"""
        return self.Mx * self.My

    @property
    def N(self) -> int:
        """返回 RIS 总反射单元数。"""
        return self.Nx * self.Ny

    @property
    def sigma2(self) -> float:
        """根据带宽和噪声参数计算等效噪声功率。"""
        noise_dbm = self.noise_density_dbm_hz + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 1e-3 * 10 ** (noise_dbm / 10.0)
