from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from datasets import DiabeticFootDataset
from losses import binary_segmentation_metrics, segmentation_loss
from models import DINOv3Backbone, MultiTaskSegModel
from paths import DEFAULT_BODY_ROOT
from paths import DEFAULT_CLOSEUP_NEGATIVE_ROOT
from paths import DEFAULT_FOOT_ROOT
from paths import DEFAULT_HUMANBODY_ROOT
from paths import DEFAULT_ULCER_ROOT
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from paths import TRAIN_OUTPUT_DIR as DEFAULT_TRAIN_OUTPUT_DIR
from training_log import (
    TrainingLogger,
    collect_dataset_info,
    collect_environment_info,
    count_model_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 multi-task DFU segmentation.")
    parser.add_argument("--foot-root", type=Path, default=DEFAULT_FOOT_ROOT)
    parser.add_argument("--body-root", type=Path, default=DEFAULT_BODY_ROOT)
    parser.add_argument(
        "--no-body",
        action="store_true",
        help="Exclude roboflow-body (natural resolution, no zoom augment).",
    )
    parser.add_argument("--humanbody-root", type=Path, default=DEFAULT_HUMANBODY_ROOT)
    parser.add_argument(
        "--no-humanbody",
        action="store_true",
        help="Exclude roboflow-humanbody (foot-only masks + body hard negatives).",
    )
    parser.add_argument("--closeup-negative-root", type=Path, default=DEFAULT_CLOSEUP_NEGATIVE_ROOT)
    parser.add_argument(
        "--no-closeup-negative",
        action="store_true",
        help="Exclude closeup-negative folder and synthetic humanbody close-up duplicates.",
    )
    parser.add_argument(
        "--negative-oversample",
        type=int,
        default=4,
        help="Repeat foot-negative training samples this many times (train only).",
    )
    parser.add_argument(
        "--neg-loss-weight",
        type=float,
        default=3.0,
        help="Per-sample loss multiplier for foot images with empty masks.",
    )
    parser.add_argument("--negative-fullbody-scale-min", type=float, default=1.2)
    parser.add_argument("--negative-fullbody-scale-max", type=float, default=1.8)
    parser.add_argument("--negative-closeup-scale-min", type=float, default=2.0)
    parser.add_argument("--negative-closeup-scale-max", type=float, default=3.5)
    parser.add_argument("--ulcer-root", type=Path, default=DEFAULT_ULCER_ROOT)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory name under --output-dir. Defaults to a timestamp when --output-dir is the default.",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--foot-augment", action="store_true")
    parser.add_argument("--foot-scale-min", type=float, default=1.5)
    parser.add_argument("--foot-scale-max", type=float, default=2.5)
    parser.add_argument("--foot-hflip-prob", type=float, default=0.5)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    task: str,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = DiabeticFootDataset(
        task=task,
        split=split,
        foot_root=args.foot_root,
        body_root=None if args.no_body else args.body_root,
        humanbody_root=None if args.no_humanbody else args.humanbody_root,
        closeup_negative_root=None if args.no_closeup_negative else args.closeup_negative_root,
        ulcer_root=args.ulcer_root,
        image_size=args.image_size,
        seed=args.seed,
        augment=bool(task == "foot" and split == "train" and args.foot_augment),
        scale_min=args.foot_scale_min,
        scale_max=args.foot_scale_max,
        hflip_prob=args.foot_hflip_prob,
        negative_oversample=args.negative_oversample if task == "foot" else 1,
        neg_sample_weight=args.neg_loss_weight if task == "foot" else 1.0,
        negative_fullbody_scale_min=args.negative_fullbody_scale_min,
        negative_fullbody_scale_max=args.negative_fullbody_scale_max,
        negative_closeup_scale_min=args.negative_closeup_scale_min,
        negative_closeup_scale_max=args.negative_closeup_scale_max,
        synthetic_closeup_from_humanbody=bool(
            task == "foot"
            and not args.no_closeup_negative
            and not args.no_humanbody
            and args.no_body
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )


def move_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    loss_weight = batch.get("loss_weight")
    if loss_weight is None:
        return images, masks, None
    return images, masks, loss_weight.to(device, non_blocking=True)


def train_one_task_batch(
    model: MultiTaskSegModel,
    batch: dict,
    task: str,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    images, masks, loss_weight = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(device, use_amp):
        outputs = model(images)
        logits = outputs[task]
        loss = segmentation_loss(logits, masks, sample_weights=loss_weight)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        metrics = binary_segmentation_metrics(logits, masks)
    return float(loss.detach().item()), metrics


def _average_metric_batches(metric_batches: list[dict[str, float]]) -> dict[str, float]:
    if not metric_batches:
        return {"dice": 0.0, "iou": 0.0, "accuracy": 0.0}
    keys = metric_batches[0].keys()
    return {key: sum(batch[key] for batch in metric_batches) / len(metric_batches) for key in keys}


def train_epoch(
    model: MultiTaskSegModel,
    foot_loader: DataLoader,
    ulcer_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
) -> dict[str, float]:
    model.train()
    foot_total = 0.0
    ulcer_total = 0.0
    foot_steps = 0
    ulcer_steps = 0
    foot_metric_batches: list[dict[str, float]] = []
    ulcer_metric_batches: list[dict[str, float]] = []

    for batch in _limited(foot_loader, limit_batches):
        loss, metrics = train_one_task_batch(
            model, batch, "foot", optimizer, scaler, device, use_amp
        )
        foot_total += loss
        foot_steps += 1
        foot_metric_batches.append(metrics)

    for batch in _limited(ulcer_loader, limit_batches):
        loss, metrics = train_one_task_batch(
            model, batch, "ulcer", optimizer, scaler, device, use_amp
        )
        ulcer_total += loss
        ulcer_steps += 1
        ulcer_metric_batches.append(metrics)

    foot_loss = foot_total / max(foot_steps, 1)
    ulcer_loss = ulcer_total / max(ulcer_steps, 1)
    foot_metrics = _average_metric_batches(foot_metric_batches)
    ulcer_metrics = _average_metric_batches(ulcer_metric_batches)
    return {
        "foot_loss": foot_loss,
        "ulcer_loss": ulcer_loss,
        "train_loss": 0.5 * (foot_loss + ulcer_loss),
        "foot_train_dice": foot_metrics["dice"],
        "foot_train_iou": foot_metrics["iou"],
        "foot_train_accuracy": foot_metrics["accuracy"],
        "ulcer_train_dice": ulcer_metrics["dice"],
        "ulcer_train_iou": ulcer_metrics["iou"],
        "ulcer_train_accuracy": ulcer_metrics["accuracy"],
        "train_dice": 0.5 * (foot_metrics["dice"] + ulcer_metrics["dice"]),
        "train_iou": 0.5 * (foot_metrics["iou"] + ulcer_metrics["iou"]),
        "train_accuracy": 0.5 * (foot_metrics["accuracy"] + ulcer_metrics["accuracy"]),
    }


@torch.no_grad()
def validate_task(
    model: MultiTaskSegModel,
    loader: DataLoader,
    task: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    steps = 0

    for batch in _limited(loader, limit_batches):
        images, masks, loss_weight = move_batch(batch, device)
        outputs = model(images)
        logits = outputs[task]
        metrics = binary_segmentation_metrics(logits, masks)
        total_loss += float(segmentation_loss(logits, masks, sample_weights=loss_weight).item())
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_accuracy += metrics["accuracy"]
        steps += 1

    denom = max(steps, 1)
    return {
        f"{task}_val_loss": total_loss / denom,
        f"{task}_val_dice": total_dice / denom,
        f"{task}_val_iou": total_iou / denom,
        f"{task}_val_accuracy": total_accuracy / denom,
        f"{task}_dice": total_dice / denom,
        f"{task}_iou": total_iou / denom,
        f"{task}_accuracy": total_accuracy / denom,
    }


def _limited(loader: Iterable, limit: int | None) -> Iterable:
    if limit is None:
        yield from loader
        return
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch


def make_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, use_amp: bool) -> Any:
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def save_checkpoint(
    path: Path,
    model: MultiTaskSegModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.run_name:
        return output_dir / args.run_name
    if output_dir.resolve() == DEFAULT_TRAIN_OUTPUT_DIR.resolve():
        return output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    foot_train = make_loader("foot", "train", args, shuffle=True)
    ulcer_train = make_loader("ulcer", "train", args, shuffle=True)
    foot_val = make_loader("foot", "val", args, shuffle=False)
    ulcer_val = make_loader("ulcer", "val", args, shuffle=False)

    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=not args.unfreeze_backbone,
    )
    model = MultiTaskSegModel(backbone=backbone).to(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(use_amp)

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_dataset_info(args, foot_train, ulcer_train, foot_val, ulcer_val),
        environment=collect_environment_info(device),
        model_info={
            "model": model.__class__.__name__,
            "backbone_frozen": not args.unfreeze_backbone,
            **count_model_parameters(model),
        },
    )
    print(f"Training logs will be saved to: {args.output_dir}")

    best_score = -1.0
    training_started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        train_metrics = train_epoch(
            model,
            foot_train,
            ulcer_train,
            optimizer,
            scaler,
            device,
            use_amp,
            args.limit_train_batches,
        )
        foot_metrics = validate_task(model, foot_val, "foot", device, args.limit_val_batches)
        ulcer_metrics = validate_task(model, ulcer_val, "ulcer", device, args.limit_val_batches)
        metrics = {**train_metrics, **foot_metrics, **ulcer_metrics}
        metrics["val_dice"] = 0.5 * (metrics["foot_val_dice"] + metrics["ulcer_val_dice"])
        metrics["val_iou"] = 0.5 * (metrics["foot_val_iou"] + metrics["ulcer_val_iou"])
        metrics["val_accuracy"] = 0.5 * (metrics["foot_val_accuracy"] + metrics["ulcer_val_accuracy"])
        score = metrics["val_dice"]
        epoch_seconds = perf_counter() - epoch_started
        is_best = score > best_score

        print(f"epoch={epoch:03d} | {format_metrics(metrics)}")
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, metrics, args)
        if is_best:
            best_score = score
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, metrics, args)

        logger.log_epoch(
            epoch=epoch,
            metrics=metrics,
            score=score,
            epoch_seconds=epoch_seconds,
            is_best=is_best,
        )

    logger.finalize(total_seconds=perf_counter() - training_started)
    print(f"Training complete. Logs and checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
