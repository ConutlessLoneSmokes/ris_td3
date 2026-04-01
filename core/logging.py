from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from core.io import write_json
from core.types import ExperimentPaths


class ExperimentLogger:
    """统一管理 CSV、TensorBoard 和 summary 输出。"""

    def __init__(
        self,
        paths: ExperimentPaths,
        train_fieldnames: list[str],
        eval_fieldnames: list[str],
    ) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=str(paths.tb_dir))

        self._train_file = paths.train_csv.open("w", encoding="utf-8", newline="")
        self._eval_file = paths.eval_csv.open("w", encoding="utf-8", newline="")

        self.train_writer = csv.DictWriter(self._train_file, fieldnames=train_fieldnames)
        self.eval_writer = csv.DictWriter(self._eval_file, fieldnames=eval_fieldnames)
        self.train_writer.writeheader()
        self.eval_writer.writeheader()

    def add_scalars(self, namespace: str, scalars: dict[str, float], step: int) -> None:
        """将一组标量写入 TensorBoard。"""
        for key, value in scalars.items():
            self.writer.add_scalar(f"{namespace}/{key}", value, step)

    def write_train_row(self, row: dict[str, object]) -> None:
        """写入一行训练指标。"""
        self.train_writer.writerow(row)

    def write_eval_row(self, row: dict[str, object]) -> None:
        """写入一行评估指标。"""
        self.eval_writer.writerow(row)

    def write_summary(self, payload: dict) -> None:
        """写入实验汇总文件。"""
        write_json(self.paths.summary_json, payload)

    def write_evaluation_summary(self, payload: dict) -> None:
        """写入独立评估汇总文件。"""
        write_json(self.paths.evaluation_summary_json, payload)

    def close(self) -> None:
        """关闭全部文件句柄。"""
        self._train_file.close()
        self._eval_file.close()
        self.writer.close()


def save_text(path: Path, text: str) -> None:
    """保存普通文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

