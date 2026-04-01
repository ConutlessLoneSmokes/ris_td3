# RIS 研究框架

该仓库已经重构为面向 **RIS-aided MISO URLLC** 的模块化研究框架，核心目标是把
**问题定义、求解器实现、运行入口、实验产物** 解耦，便于后续持续接入 `TD3`、`DDPG`、
随机搜索以及传统优化方法。

## 目录结构

```text
ris_td3/
├─ core/                       # 通用类型、日志、注册器、随机种子、IO 工具
├─ problems/
│  └─ ris_miso_urllc/          # 共享问题定义：配置、信道、约束、目标、评估、RL 适配器
├─ solvers/
│  ├─ rl/                      # RL 求解器：TD3、DDPG、网络、回放缓存
│  ├─ baselines/               # 基线方法：random_search
│  └─ optimization/            # 传统优化方法预留目录
├─ runners/                    # 统一训练、评估、benchmark、绘图、sanity check
├─ scripts/                    # 面向用户的薄脚本入口
├─ configs/                    # 默认配置与注册入口
└─ outputs/                    # 实验产物目录
```

## 统一接口

问题层统一对象：
- `ProblemConfig`
- `ProblemInstance`
- `Solution`
- `Metrics`
- `Evaluator`
- `ScenarioSampler`

求解器层统一接口：
- `Solver.setup(problem_config)`
- `Solver.solve(instance) -> Solution`
- `Solver.save(path)`
- `Solver.load(path)`

RL 求解器扩展接口：
- `bind_environment(state_dim, action_dim)`
- `select_action(state)`
- `update(replay_buffer)`

## 输出目录规范

所有实验结果统一写入：

```text
outputs/
  <problem_name>/
    <solver_name>/
      <run_name>/
        config.problem.json
        config.solver.json
        train_metrics.csv
        eval_metrics.csv
        summary.json
        evaluation_summary.json
        checkpoints/
        tb/
        plots/
```

默认问题名：`ris_miso_urllc`

## 常用命令

### 1. 冒烟检查

```bash
python scripts/sanity_check.py
```

### 2. 训练 TD3

```bash
python scripts/train.py
```

或显式指定：

```bash
python scripts/train_td3.py
```

### 3. 训练 DDPG

```bash
python scripts/train_ddpg.py
```

### 4. 运行随机搜索基线

```bash
python scripts/train_random_search.py --train-episodes 200 --eval-episodes 20 --num-candidates 128
```

### 5. 评估实验结果

默认评估最近一次 TD3 运行：

```bash
python scripts/evaluate.py
```

显式指定实验目录：

```bash
python scripts/evaluate.py --run-dir outputs/ris_miso_urllc/td3/<run_name>
```

### 6. 运行 benchmark

```bash
python scripts/benchmark.py --solvers td3,ddpg,random_search --eval-episodes 20
```

### 7. 绘制实验曲线

默认读取最近一次 TD3 运行：

```bash
python scripts/plot_results.py
```

显式指定实验目录：

```bash
python scripts/plot_results.py --run-dir outputs/ris_miso_urllc/td3/<run_name>
```

## 当前已接入方法

- `td3`
  - 双 Q、目标策略平滑、延迟策略更新
- `ddpg`
  - 单 Q 连续控制基线
- `random_search`
  - 直接在可行域采样候选解并择优

## 配置约定

- `problems/ris_miso_urllc/config.py`
  - 维护共享问题配置：几何位置、噪声、功率预算、RIS 参数、CBL 约束等
- `solvers/*`
  - 各求解器维护自己的超参数配置
- `configs/default.py`
  - 负责默认问题、默认求解器与内建注册

## 扩展方向

后续新增方法时，优先遵循以下边界：
- 不修改问题层公式，新增求解器放到 `solvers/`
- 统一通过 `runners/` 输出日志、checkpoint 和图像
- RL 方法使用 `problems/ris_miso_urllc/rl_env.py`
- 非 RL 方法直接走 `ProblemInstance -> Solver -> Evaluator`
