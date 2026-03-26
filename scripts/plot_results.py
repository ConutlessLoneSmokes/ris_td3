from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接从项目根目录调用脚本。
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

from configs.default import SystemConfig


def configure_fonts() -> None:
    """配置英文 Times New Roman 与中文微软雅黑字体。"""
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


def resolve_run_dir(run_dir_arg: str | None, outputs_root: str) -> Path:
    """解析用户指定的实验目录，或自动读取最新实验目录。"""
    if run_dir_arg is not None:
        return Path(run_dir_arg)

    latest_run_file = Path(outputs_root) / "latest_run.txt"
    if not latest_run_file.exists():
        raise FileNotFoundError("未找到 latest_run.txt，请先运行训练脚本或显式传入 --run-dir。")
    return Path(latest_run_file.read_text(encoding="utf-8").strip())


def load_csv_rows(path: Path) -> list[dict[str, float]]:
    """读取 CSV 文件并将可用字段转换为浮点数。"""
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


def main() -> None:
    """读取训练日志并绘制学习曲线。"""
    parser = argparse.ArgumentParser(description="绘制训练与评估曲线")
    parser.add_argument("--run-dir", type=str, default=None, help="实验目录，默认读取最新实验目录")
    parser.add_argument("--window", type=int, default=100, help="训练曲线滑动平均窗口")
    args = parser.parse_args()

    cfg = SystemConfig()
    configure_fonts()
    run_dir = resolve_run_dir(args.run_dir, cfg.outputs_root)
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
        axes[0].plot(
            train_episode,
            moving_average(train_reward, args.window),
            color="#1565C0",
            linewidth=2.0,
            label=f"滑动平均({args.window})",
        )
        axes[0].set_title("训练回合 Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(train_episode, train_sinr, color="#2E7D32", linewidth=1.5)
        axes[1].set_title("训练回合 Mean SINR")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Mean SINR")
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

        axes[2].plot(eval_episode, eval_reward, marker="o", color="#6A1B9A", linewidth=1.8)
        axes[2].plot(eval_episode, eval_sinr, marker="s", color="#00838F", linewidth=1.4)
        axes[2].set_title("评估指标")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Value")
        axes[2].grid(alpha=0.3)
        axes[2].legend(["avg_reward", "avg_mean_sinr"])

    fig.suptitle(f"RIS-TD3 实验曲线: {run_dir.name}")
    fig.tight_layout()

    output_path = plots_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"实验目录: {run_dir}")
    print(f"图像已保存: {output_path}")


if __name__ == "__main__":
    main()
