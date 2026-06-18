from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.loaders import make_loader as make_segmentation_loader
from losses import binary_segmentation_metrics, segmentation_loss
from models import DINOv3Backbone, FastInstFootHead, FastInstWoundHead
from training_log import (
    TrainingLogger,
    collect_dataset_stats,
    collect_environment_info,
    count_model_parameters,
)
from trainers.common import format_metrics
from trainers.common import limited_batches
from trainers.common import make_lr_scheduler
from utils.runtime import autocast_context
from utils.runtime import make_grad_scaler


def build_segmentation_head(task: str) -> nn.Module:
    if task == "foot":
        return FastInstFootHead()
    if task == "wound":
        return FastInstWoundHead()
    raise ValueError(f"Unsupported segmentation task: {task}")


def build_segmentation_train_stack(
    args: argparse.Namespace,
    task: str,
    device: torch.device,
) -> tuple[DINOv3Backbone, nn.Module]:
    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=not args.unfreeze_backbone,
    ).to(device)
    head = build_segmentation_head(task).to(device)
    return backbone, head


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


def predict_segmentation_logits(
    backbone: DINOv3Backbone,
    head: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    features = backbone(images)
    output_size = tuple(int(value) for value in images.shape[-2:])
    logits = head(features)
    return F.interpolate(
        logits,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )


def train_one_batch(
    backbone: DINOv3Backbone,
    head: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    images, masks, loss_weight = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(device, use_amp):
        logits = predict_segmentation_logits(backbone, head, images)
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
    backbone: DINOv3Backbone,
    head: nn.Module,
    loader: DataLoader,
    task: str,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
) -> dict[str, float]:
    backbone.eval()
    head.train()
    total = 0.0
    steps = 0
    metric_batches: list[dict[str, float]] = []

    for batch in limited_batches(loader, limit_batches):
        loss, metrics = train_one_batch(
            backbone, head, batch, optimizer, scaler, device, use_amp
        )
        total += loss
        steps += 1
        metric_batches.append(metrics)

    loss = total / max(steps, 1)
    task_metrics = _average_metric_batches(metric_batches)
    return {
        "train_loss": loss,
        f"{task}_loss": loss,
        "train_dice": task_metrics["dice"],
        "train_iou": task_metrics["iou"],
        "train_accuracy": task_metrics["accuracy"],
        f"{task}_train_dice": task_metrics["dice"],
        f"{task}_train_iou": task_metrics["iou"],
        f"{task}_train_accuracy": task_metrics["accuracy"],
    }


@torch.no_grad()
def validate_task(
    backbone: DINOv3Backbone,
    head: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, float]:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    steps = 0

    for batch in limited_batches(loader, limit_batches):
        images, masks, loss_weight = move_batch(batch, device)
        logits = predict_segmentation_logits(backbone, head, images)
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


def save_checkpoint(
    path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task": args.task,
        "epoch": epoch,
        "head_state_dict": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def collect_task_dataset_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "splits": {
            "train": collect_dataset_stats(train_loader.dataset, train_loader),
            "val": collect_dataset_stats(val_loader.dataset, val_loader),
        },
        "total_train_samples": len(train_loader.dataset),
        "total_val_samples": len(val_loader.dataset),
    }


def run_segmentation_training(
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    *,
    task: str,
) -> None:
    args.task = task
    train_loader = make_segmentation_loader(task, "train", args, shuffle=True)
    val_loader = make_segmentation_loader(task, "val", args, shuffle=False)

    backbone, head = build_segmentation_train_stack(args, task, device)
    trainable_params = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if args.unfreeze_backbone:
        trainable_params.extend(
            parameter for parameter in backbone.parameters() if parameter.requires_grad
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args)
    scaler = make_grad_scaler(use_amp)

    logger = TrainingLogger(args.output_dir)
    logger.write_initial_artifacts(
        args=args,
        dataset_info=collect_task_dataset_info(train_loader, val_loader),
        environment=collect_environment_info(device),
        model_info={
            "backbone": backbone.__class__.__name__,
            "head": head.__class__.__name__,
            "task": task,
            "backbone_frozen": not args.unfreeze_backbone,
            **count_model_parameters(head),
        },
    )
    print(f"Training logs will be saved to: {args.output_dir}")

    best_score = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    training_started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        train_metrics = train_epoch(
            backbone,
            head,
            train_loader,
            task,
            optimizer,
            scaler,
            device,
            use_amp,
            args.limit_train_batches,
        )
        task_metrics = validate_task(backbone, head, val_loader, task, device, args.limit_val_batches)
        metrics = {**train_metrics, **task_metrics}
        metrics["val_dice"] = metrics[f"{task}_val_dice"]
        metrics["val_iou"] = metrics[f"{task}_val_iou"]
        metrics["val_accuracy"] = metrics[f"{task}_val_accuracy"]
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        score = metrics["val_dice"]
        epoch_seconds = perf_counter() - epoch_started
        is_best = score > best_score

        display_metrics = {key: value for key, value in metrics.items() if key != "learning_rate"}
        print(
            f"epoch={epoch:03d} | lr={metrics['learning_rate']:.2e} | {format_metrics(display_metrics)}"
        )
        save_checkpoint(
            args.output_dir / "last.pt", head, optimizer, epoch, metrics, args, scheduler
        )
        if is_best:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best.pt", head, optimizer, epoch, metrics, args, scheduler
            )
        else:
            epochs_without_improvement += 1

        logger.log_epoch(
            epoch=epoch,
            metrics=metrics,
            score=score,
            epoch_seconds=epoch_seconds,
            is_best=is_best,
        )

        if scheduler is not None:
            scheduler.step()

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_dice improvement for {args.early_stopping_patience} epochs "
                f"(best epoch {logger.best_epoch}, score {logger.best_score:.4f})"
            )
            break

    logger.finalize(
        total_seconds=perf_counter() - training_started,
        early_stopping={
            "enabled": args.early_stopping_patience > 0,
            "patience": args.early_stopping_patience,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
    )
    if stopped_early:
        print(
            f"Training stopped early at epoch {stop_epoch}. "
            f"Best checkpoint: epoch {logger.best_epoch} (val_dice={logger.best_score:.4f})"
        )
    print(f"Training complete. Logs and checkpoints saved to: {args.output_dir}")
