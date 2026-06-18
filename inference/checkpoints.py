from __future__ import annotations

import warnings
from pathlib import Path

import torch

from models import DFUPipelineModel, DINOv3Backbone
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO

DEFAULT_IMAGE_SIZE = 384


def checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like torch payload.")
    for key in ("head_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain head_state_dict or state_dict weights.")


def strip_first_matching_prefix(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    for prefix in prefixes:
        prefixed = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if prefixed:
            return prefixed
    return state_dict


def load_segmentation_head(
    head: torch.nn.Module,
    checkpoint_path: Path,
    *,
    prefixes: tuple[str, ...],
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_first_matching_prefix(checkpoint_state_dict(checkpoint), prefixes)
    missing_keys, unexpected_keys = head.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        warnings.warn(
            f"{checkpoint_path} loaded with non-strict head key mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )


def load_pipeline_model(
    *,
    foot_head_checkpoint: Path | None,
    wound_head_checkpoint: Path | None,
    dinov3_repo: Path = DEFAULT_DINOV3_REPO,
    dinov3_checkpoint: Path = DEFAULT_DINOV3_CHECKPOINT,
    device: torch.device,
) -> DFUPipelineModel:
    backbone = DINOv3Backbone(
        repo_dir=dinov3_repo,
        checkpoint_path=dinov3_checkpoint,
        freeze=True,
    )
    model = DFUPipelineModel(backbone=backbone).to(device)
    if foot_head_checkpoint is not None:
        load_segmentation_head(
            model.foot_head,
            foot_head_checkpoint,
            prefixes=("foot_head.", "head."),
            device=device,
        )
    if wound_head_checkpoint is not None:
        load_segmentation_head(
            model.wound_head,
            wound_head_checkpoint,
            prefixes=("wound_head.", "ulcer_head.", "head."),
            device=device,
        )
    model.eval()
    return model


def resolve_image_size_from_checkpoint(
    checkpoint_path: Path,
    requested_image_size: int | None,
) -> int:
    if requested_image_size is not None:
        return int(requested_image_size)
    if checkpoint_path is None:
        return DEFAULT_IMAGE_SIZE
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        warnings.warn(
            f"Could not read image_size from checkpoint ({exc}); using {DEFAULT_IMAGE_SIZE}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_IMAGE_SIZE

    if isinstance(checkpoint, dict):
        args = checkpoint.get("args")
        if isinstance(args, dict) and args.get("image_size") is not None:
            return int(args["image_size"])
    return DEFAULT_IMAGE_SIZE
