from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

from configs.default import DEFAULT_PROBLEM_NAME, DEFAULT_SOLVER_NAME, build_problem_config
from core.io import resolve_latest_run


def configure_fonts() -> None:
    """配置英文字体为 Times New Roman，中文字体为微软雅黑。"""
    candidate_font_files = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
    ]
    for font_file in candidate_font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))

    rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
    rcParams["axes.unicode_minus"] = False


def parse_args(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> argparse.Namespace:
    """解析画图脚本参数。"""
    parser = argparse.ArgumentParser(description="统一实验曲线绘图入口")
    parser.add_argument("--problem", type=str, default=default_problem, help="问题名称")
    parser.add_argument("--solver", type=str, default=default_solver, help="求解器名称")
    parser.add_argument("--run-dir", type=str, default=None, help="显式指定实验目录")
    parser.add_argument("--window", type=int, default=100, help="训练曲线滑动平均窗口")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, float]]:
    """读取 CSV 文件并将可用字段转成浮点数。"""
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算滑动平均，用于平滑训练曲线。"""
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def main(default_problem: str = DEFAULT_PROBLEM_NAME, default_solver: str = DEFAULT_SOLVER_NAME) -> None:
    args = parse_args(default_problem=default_problem, default_solver=default_solver)
    problem_config = build_problem_config(args.problem)
    configure_fonts()

    run_dir = Path(args.run_dir) if args.run_dir is not None else resolve_latest_run(problem_config.outputs_root, args.problem, args.solver)
    if run_dir is None:
        raise FileNotFoundError("未找到最新实验目录，请先训练或显式指定 --run-dir。")

    train_rows = load_csv_rows(run_dir / "train_metrics.csv")
    eval_rows = load_csv_rows(run_dir / "eval_metrics.csv")
    if not train_rows and not eval_rows:
        raise FileNotFoundError(f"在 {run_dir} 下未找到 train_metrics.csv 或 eval_metrics.csv")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    if train_rows:
        train_episode = np.array([row["episode"] for row in train_rows], dtype=np.float64)
        train_reward = np.array([row["reward"] for row in train_rows], dtype=np.float64)
        train_sinr = np.array([row["mean_sinr"] for row in train_rows], dtype=np.float64)
        train_power = np.array([row["total_power"] for row in train_rows], dtype=np.float64)

        axes[0].plot(train_episode, train_reward, color="#B0BEC5", linewidth=1.0, label="原始 reward")
        axes[0].plot(train_episode, moving_average(train_reward, args.window), color="#1565C0", linewidth=2.0, label=f"滑动平均({args.window})")
        axes[0].set_title("训练回合 Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(train_episode, train_sinr, color="#2E7D32", linewidth=1.5)
        axes[1].set_title("训练回合 Mean SINR")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Mean SINR")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1].grid(alpha=0.3)

        axes[3].plot(train_episode, train_power, color="#EF6C00", linewidth=1.5)
        axes[3].set_title("训练回合总发射功率")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Power")
        axes[3].grid(alpha=0.3)

    if eval_rows:
        eval_episode = np.array([row["episode"] for row in eval_rows], dtype=np.float64)
        eval_reward = np.array([row["avg_reward"] for row in eval_rows], dtype=np.float64)
        eval_sinr = np.array([row["avg_mean_sinr"] for row in eval_rows], dtype=np.float64)

        axes[2].plot(eval_episode, eval_reward, marker="o", color="#6A1B9A", linewidth=1.8, label="avg_reward")
        axes[2].set_title("评估 Reward")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Reward")
        axes[2].grid(alpha=0.3)

        ax2 = axes[2].twinx()
        ax2.plot(eval_episode, eval_sinr, marker="s", color="#00838F", linewidth=1.4, label="avg_mean_sinr")
        ax2.set_ylabel("Mean SINR")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        lines_1, labels_1 = axes[2].get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.suptitle(f"{args.problem}/{args.solver} 实验曲线: {run_dir.name}")
    fig.tight_layout()

    output_path = plots_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"实验目录: {run_dir}")
    print(f"图像已保存: {output_path}")


if __name__ == "__main__":
    main()
