from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _per_sample_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_per_sample = bce_map.flatten(1).mean(dim=1)
    dice_per_sample = _per_sample_dice_loss(logits, targets)

    if sample_weights is None:
        bce = bce_per_sample.mean()
        dice = dice_per_sample.mean()
    else:
        weights = sample_weights.to(dtype=bce_per_sample.dtype, device=bce_per_sample.device)
        weight_sum = weights.sum().clamp_min(1.0e-6)
        bce = (bce_per_sample * weights).sum() / weight_sum
        dice = (dice_per_sample * weights).sum() / weight_sum

    return bce_weight * bce + dice_weight * dice


@torch.no_grad()
def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = (targets > 0.5).float()

    preds_flat = preds.flatten(1)
    targets_flat = targets.flatten(1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    pred_sum = preds_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
    }
