from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets import ClassificationImageDataset, DiabeticFootDataset
from eval.sample_loaders import load_foot_test_samples, load_wound_test_samples
from inference.checkpoints import load_segmentation_head, resolve_image_size_from_checkpoint
from models import DINOv3Backbone, DFUFeatureClassifierHead
from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from trainers.dfu_trainer import DFU_CLASSES, validate_dfu
from trainers.segmentation import build_segmentation_head, validate_task
from utils.runtime import resolve_device


def _make_segmentation_loader(
    task: str,
    samples: list,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = DiabeticFootDataset(
        task=task,
        split="val",
        image_size=image_size,
        samples=samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _load_segmentation_stack(
    task: str,
    checkpoint: Path,
    *,
    dinov3_model: Path,
    device: torch.device,
) -> tuple[DINOv3Backbone, torch.nn.Module]:
    prefixes = ("foot_head.", "head.") if task == "foot" else ("wound_head.", "ulcer_head.", "head.")
    backbone = DINOv3Backbone(
        model_path=dinov3_model,
        freeze=True,
    ).to(device)
    head = build_segmentation_head(task).to(device)
    load_segmentation_head(head, checkpoint, prefixes=prefixes, device=device)
    backbone.eval()
    head.eval()
    return backbone, head


def evaluate_foot(
    *,
    checkpoint: Path,
    data_root: Path,
    image_size: int | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    dinov3_model: Path = DEFAULT_DINOV3_MODEL_PATH,
    device_name: str = "auto",
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    image_size = resolve_image_size_from_checkpoint(checkpoint, image_size)
    samples = load_foot_test_samples(data_root)
    loader = _make_segmentation_loader(
        "foot",
        samples,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    backbone, head = _load_segmentation_stack(
        "foot",
        checkpoint,
        dinov3_model=dinov3_model,
        device=device,
    )
    metrics = validate_task(backbone, head, loader, "foot", device, limit_batches)
    return {
        "task": "foot",
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "image_size": image_size,
        "num_samples": len(samples),
        "metrics": metrics,
        "amp": bool(amp and device.type == "cuda"),
    }


def evaluate_wound(
    *,
    checkpoint: Path,
    image_dir: Path,
    mask_dir: Path,
    image_size: int | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    dinov3_model: Path = DEFAULT_DINOV3_MODEL_PATH,
    device_name: str = "auto",
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    image_size = resolve_image_size_from_checkpoint(checkpoint, image_size)
    samples = load_wound_test_samples(image_dir, mask_dir)
    if not samples:
        raise RuntimeError("No wound image/mask pairs found for evaluation.")
    loader = _make_segmentation_loader(
        "wound",
        samples,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    backbone, head = _load_segmentation_stack(
        "wound",
        checkpoint,
        dinov3_model=dinov3_model,
        device=device,
    )
    metrics = validate_task(backbone, head, loader, "wound", device, limit_batches)
    return {
        "task": "wound",
        "checkpoint": str(checkpoint),
        "image_size": image_size,
        "num_samples": len(samples),
        "metrics": metrics,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "amp": bool(amp and device.type == "cuda"),
    }


def _load_dfu_stack(
    checkpoint: Path,
    *,
    dinov3_model: Path,
    device: torch.device,
) -> tuple[DINOv3Backbone, DFUFeatureClassifierHead]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"DFU checkpoint must be a dict: {checkpoint}")

    classes = tuple(payload.get("classes", DFU_CLASSES))
    feature_dim = int(payload.get("feature_dim", 384))
    hidden_dim = int(payload.get("hidden_dim", 256))
    dropout = float(payload.get("dropout", 0.2))
    head_type = str(payload.get("head_type", "linear"))

    backbone = DINOv3Backbone(
        model_path=dinov3_model,
        freeze=True,
    ).to(device)
    head = DFUFeatureClassifierHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        dropout=dropout,
        head_type=head_type,
    ).to(device)
    load_segmentation_head(
        head,
        checkpoint,
        prefixes=("classification_head.", "dfu_head.", "head."),
        device=device,
    )
    backbone.eval()
    head.eval()
    return backbone, head


def evaluate_dfu(
    *,
    checkpoint: Path,
    data_root: Path,
    image_size: int | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    dinov3_model: Path = DEFAULT_DINOV3_MODEL_PATH,
    device_name: str = "auto",
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    image_size = resolve_image_size_from_checkpoint(checkpoint, image_size)
    # data_root should point directly to the ImageFolder root (e.g. .../test with dfu/other/).
    dataset = ClassificationImageDataset(
        root=data_root,
        split="train",
        image_size=image_size,
        classes=DFU_CLASSES,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    backbone, head = _load_dfu_stack(
        checkpoint,
        dinov3_model=dinov3_model,
        device=device,
    )
    use_amp = bool(amp and device.type == "cuda")
    metrics = validate_dfu(backbone, head, loader, device, use_amp, None, limit_batches)
    return {
        "task": "dfu",
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "image_size": image_size,
        "num_samples": len(dataset),
        "class_counts": {
            dataset.id2label[index]: sum(1 for sample in dataset.samples if sample.label == index)
            for index in range(len(dataset.classes))
        },
        "metrics": metrics,
        "amp": use_amp,
    }


def save_eval_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
