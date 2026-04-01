from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class ProblemConfig:
    """RIS-aided MISO URLLC 共享问题配置。"""

    name: str = "ris_miso_urllc"  # 问题名称，用于输出目录与注册器

    # 系统拓扑参数
    K: int = 4  # 用户数量
    Mx: int = 2  # 基站 UPA 在 x 方向的阵元数
    My: int = 2  # 基站 UPA 在 y 方向的阵元数
    Nx: int = 4  # RIS 在 x 方向的单元数
    Ny: int = 4  # RIS 在 y 方向的单元数

    # 无线物理层参数
    bandwidth_hz: float = 1e5  # 系统带宽
    noise_density_dbm_hz: float = -174.0  # 噪声功率谱密度
    noise_figure_db: float = 3.0  # 噪声系数
    p_total_watt: float = 1e-3  # 基站总发射功率预算
    wavelength: float = 0.1  # 载波波长
    element_spacing: float = 0.05  # 阵元间距

    # 有限块长通信参数
    total_cbl: float = 100.0  # 总块长预算
    min_cbl: float = 10.0  # 每用户最小块长
    target_error_prob: float = 1e-12  # 目标误块率

    # Rician 信道参数
    zeta_br: float = 10.0  # BS-RIS 链路 Rician 因子
    zeta_ru: float = 10.0  # RIS-用户链路 Rician 因子

    # 路损模型参数
    pl0_db: float = -30.0  # 参考路损常数
    pathloss_exp: float = 2.2  # 路损指数

    # 二维位置
    bs_pos: List[float] = field(default_factory=lambda: [0.0, 0.0])  # 基站平面位置
    ris_pos: List[float] = field(default_factory=lambda: [40.0, 0.0])  # RIS 平面位置
    user_pos: List[List[float]] = field(
        default_factory=lambda: [
            [16.0, 40.0],
            [32.0, 40.0],
            [48.0, 40.0],
            [64.0, 40.0],
        ]
    )  # 用户平面位置

    # 三维高度
    bs_height: float = 12.5  # 基站高度
    ris_height: float = 12.5  # RIS 高度
    user_height: float = 1.5  # 用户高度

    # RIS 实际幅相耦合参数
    beta_min: float = 0.4  # 最小反射幅度
    alpha_ris: float = 1.9  # 幅相曲线陡峭程度
    phi_ris: float = 0.43 * math.pi  # 幅相曲线偏移量

    # 通用实验控制参数
    seed: int = 42  # 全局随机种子
    eps_div: float = 1e-30  # 防止除零的小常数
    train_episodes: int = 10000  # 统一训练/运行回合数
    eval_episodes: int = 100  # 统一评估回合数
    eval_interval: int = 500  # 训练时的周期性评估间隔
    save_interval: int = 1000  # checkpoint 保存间隔
    max_steps: int = 100  # 单个场景下的最大优化步数
    outputs_root: str = "outputs"  # 全部实验输出根目录

    @property
    def M(self) -> int:
        """返回基站总天线数。"""
        return self.Mx * self.My

    @property
    def N(self) -> int:
        """返回 RIS 总单元数。"""
        return self.Nx * self.Ny

    @property
    def sigma2(self) -> float:
        """根据带宽与噪声参数计算等效噪声功率。"""
        noise_dbm = self.noise_density_dbm_hz + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 1e-3 * 10 ** (noise_dbm / 10.0)

