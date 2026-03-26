# RIS-TD3 论文复现

基于论文 *Deep Reinforcement Learning for Practical Phase Shift Optimization in RIS-aided MISO URLLC Systems* 的复现工程，当前已经完成信道/环境建模、TD3 训练主链路、实验记录归档、评估脚本和基础画图脚本。

## 当前实现状态

- 已完成：
  - 信道建模、有限块长 FBL 奖励、RIS 实际幅相关系、动作约束映射
  - 单步 single-shot `RISEnv`
  - 论文对齐版 TD3 网络、经验回放、训练与评估
  - TensorBoard、CSV、JSON 的实验记录保存
  - 训练后结果画图脚本
- 当前默认对齐论文的参数：
  - actor 学习率 `1e-4`
  - critic 学习率 `1e-4`
  - actor 隐层 `[800, 400, 200]`
  - critic 结构：状态分支 `800`、动作分支 `800`、融合后 `[600, 400]`
  - `LayerNorm = True`
  - replay buffer `10000`
  - batch size `64`
  - exploration noise 方差 `0.1`
  - target policy noise 方差 `0.1`
  - `tau = 0.005`
  - `policy_delay = 4`
  - `eval_episodes = 100`

## 与论文的已知差异

- `gamma`：论文正文没有给出具体数值，当前保留 `0.99`。由于当前环境为单步回合，`done=True`，训练中未来回报项实际上不生效。
- `noise_clip`：论文给出了 TD3 中的截断常数 `c`，但未报告具体数值，当前采用常见默认值 `0.5`。
- `warmup_episodes`：论文算法描述没有单独给出随机 warmup 阶段，当前默认设为 `0`。

## 目录结构

```text
ris_td3/
├─ agents/
│  ├─ networks.py
│  ├─ replay_buffer.py
│  └─ td3.py
├─ configs/
│  └─ default.py
├─ envs/
│  ├─ channel_model.py
│  ├─ constraints.py
│  ├─ fbl.py
│  └─ ris_env.py
├─ scripts/
│  ├─ evaluate.py
│  ├─ plot_results.py
│  ├─ sanity_check.py
│  └─ train.py
└─ outputs/
   └─ experiments/
      ├─ latest_run.txt
      └─ <run_name>/
         ├─ config.json
         ├─ summary.json
         ├─ evaluation_summary.json
         ├─ train_metrics.csv
         ├─ eval_metrics.csv
         ├─ tb/
         ├─ checkpoints/
         │  ├─ latest.pt
         │  └─ best.pt
         └─ plots/
            └─ learning_curves.png
```

## 运行方式

### 1. 冒烟测试

```bash
python scripts/sanity_check.py
```

### 2. 按论文默认参数训练

```bash
python scripts/train.py
```

### 3. 小规模快速冒烟

```bash
python scripts/train.py --train-episodes 20 --batch-size 4 --buffer-size 64 --eval-interval 10 --save-interval 10 --eval-episodes 5
```

### 4. 评估最新实验

```bash
python scripts/evaluate.py
```

也可以指定模型：

```bash
python scripts/evaluate.py --checkpoint outputs/experiments/<run_name>/checkpoints/best.pt
```

### 5. 绘制训练曲线

```bash
python scripts/plot_results.py
```

也可以指定实验目录：

```bash
python scripts/plot_results.py --run-dir outputs/experiments/<run_name>
```

## 实验记录说明

每次运行 `train.py` 会在 `outputs/experiments/` 下生成一个独立实验目录，并自动写入：

- `config.json`
  - 训练配置快照和论文对齐说明
- `train_metrics.csv`
  - 每个 episode 的训练指标
- `eval_metrics.csv`
  - 周期评估指标
- `summary.json`
  - 本次训练的核心摘要、checkpoint 路径和日志路径
- `tb/`
  - TensorBoard 事件文件
- `checkpoints/`
  - `latest.pt` 和 `best.pt`

运行 `evaluate.py` 后会在对应实验目录下补充：

- `evaluation_summary.json`
  - 独立评估结果摘要

运行 `plot_results.py` 后会在对应实验目录下补充：

- `plots/learning_curves.png`
  - 训练与评估曲线图

## 后续建议

- 把论文图 3、图 4、图 7 的实验横轴与当前脚本中的 episode 数逐个对齐，补场景扫描脚本。
- 增加 `beta_min`、`p_total_watt`、`N` 的批量实验入口，直接产出论文对比曲线。
- 若后续希望进一步压论文结构，可把 deterministic policy 与 Gaussian policy 分支显式拆成两个训练配置。
