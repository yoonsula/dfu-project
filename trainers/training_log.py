from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cli.dataset_args import foot_roots_for_args
from datasets import DiabeticFootDataset


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serialize_args(args: Any) -> dict[str, Any]:
    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def collect_dataset_stats(dataset: DiabeticFootDataset, loader: DataLoader) -> dict[str, Any]:
    positives = sum(1 for sample in dataset.samples if not sample.is_negative)
    negatives = sum(1 for sample in dataset.samples if sample.is_negative)
    return {
        "task": dataset.task,
        "split": dataset.split,
        "count": len(dataset),
        "positive_count": positives,
        "negative_count": negatives,
        "batch_size": loader.batch_size,
        "num_batches": len(loader),
    }


def collect_dataset_info(
    args: Any,
    foot_train: DataLoader,
    wound_train: DataLoader,
    foot_val: DataLoader,
    wound_val: DataLoader,
) -> dict[str, Any]:
    return {
        "paths": {
            "foot_roots": [
                str(path)
                for path in foot_roots_for_args(args)
            ],
            "body_root": str(args.body_root),
            "humanbody_root": str(args.humanbody_root),
            "wound_root": str(args.wound_root),
            "wound_image_root": None if args.no_wound_image else str(args.wound_image_root),
            "dinov3_model": str(args.dinov3_model),
        },
        "splits": {
            "foot_train": collect_dataset_stats(foot_train.dataset, foot_train),
            "foot_val": collect_dataset_stats(foot_val.dataset, foot_val),
            "wound_train": collect_dataset_stats(wound_train.dataset, wound_train),
            "wound_val": collect_dataset_stats(wound_val.dataset, wound_val),
        },
        "total_train_samples": len(foot_train.dataset) + len(wound_train.dataset),
        "total_val_samples": len(foot_val.dataset) + len(wound_val.dataset),
    }


def collect_environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


@dataclass
class TrainingLogger:
    output_dir: Path
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = -1.0
    best_metrics: dict[str, float] = field(default_factory=dict)
    finished: bool = False
    finished_at: str | None = None
    total_seconds: float | None = None
    early_stopping: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_initial_artifacts(
        self,
        args: Any,
        dataset_info: dict[str, Any],
        environment: dict[str, Any],
        model_info: dict[str, Any],
    ) -> None:
        self.config = _serialize_args(args)
        self.dataset = dataset_info
        self.environment = environment
        self.model = model_info
        self._flush()

    def log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        score: float,
        epoch_seconds: float,
        is_best: bool,
    ) -> None:
        self.epochs.append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "score": score,
                "epoch_seconds": round(epoch_seconds, 2),
                "is_best": is_best,
            }
        )
        if is_best:
            self.best_epoch = epoch
            self.best_score = score
            self.best_metrics = dict(metrics)
        self._flush()

    def finalize(
        self,
        total_seconds: float,
        early_stopping: dict[str, Any] | None = None,
    ) -> None:
        self.finished = True
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.total_seconds = round(total_seconds, 2)
        self.early_stopping = early_stopping or {}
        self._flush()

    def _flush(self) -> None:
        payload = {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "finished": self.finished,
            "total_seconds": self.total_seconds,
            "output_dir": str(self.output_dir),
            "config": self.config,
            "dataset": self.dataset,
            "environment": self.environment,
            "model": self.model,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "early_stopping": self.early_stopping,
        }
        path = self.output_dir / "train_log.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
