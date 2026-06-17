from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from datasets.diabetic_foot_dataset import IMAGENET_MEAN, IMAGENET_STD
from models import MultiTaskSegModel
from utils.runtime import synchronize_if_needed


@dataclass(frozen=True)
class SegmentationConfig:
    image_size: int
    foot_threshold: float = 0.5
    ulcer_threshold: float = 0.5
    min_foot_ratio: float = 0.08
    max_foot_ratio: float = 0.5
    center_tolerance: float = 0.25
    min_ulcer_ratio: float = 0.001


@dataclass(frozen=True)
class StagedSegmentationOutput:
    features: torch.Tensor
    foot_logits: torch.Tensor
    ulcer_logits: torch.Tensor | None
    backbone_ms: float
    foot_head_ms: float
    ulcer_head_ms: float


@dataclass(frozen=True)
class GatedSegmentationResult:
    foot_mask: np.ndarray
    ulcer_mask: np.ndarray
    foot_mask_small: np.ndarray
    ulcer_mask_small: np.ndarray | None
    foot_detected: bool
    foot_area_ratio: float
    foot_centered: bool
    foot_center_x: float | None
    foot_center_y: float | None
    capture_guidance: str
    ulcer_enabled: bool
    ulcer_detected: bool
    ulcer_area_ratio: float
    preprocess_ms: float
    backbone_ms: float
    foot_head_ms: float
    model_ms: float
    ulcer_head_ms: float
    postprocess_ms: float


def preprocess_image(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


@torch.inference_mode()
def forward_segmentation_staged(
    model: MultiTaskSegModel,
    input_tensor: torch.Tensor,
    device: torch.device,
    *,
    run_ulcer: bool,
    autocast_context: Callable[[], Any] | None = None,
) -> StagedSegmentationOutput:
    """Run backbone once, foot head, then optionally ulcer head on the same features."""
    output_size = tuple(int(value) for value in input_tensor.shape[-2:])
    context_factory = autocast_context or (lambda: nullcontext())

    with context_factory():
        backbone_start = perf_counter()
        features = model.encode(input_tensor)
        synchronize_if_needed(device)
        backbone_ms = (perf_counter() - backbone_start) * 1000.0

        foot_start = perf_counter()
        foot_logits = model.predict_foot_logits(features, output_size)
        synchronize_if_needed(device)
        foot_head_ms = (perf_counter() - foot_start) * 1000.0

    ulcer_logits = None
    ulcer_head_ms = 0.0
    if run_ulcer:
        with context_factory():
            ulcer_start = perf_counter()
            ulcer_logits = model.predict_ulcer_logits(features, output_size)
            synchronize_if_needed(device)
            ulcer_head_ms = (perf_counter() - ulcer_start) * 1000.0

    return StagedSegmentationOutput(
        features=features,
        foot_logits=foot_logits,
        ulcer_logits=ulcer_logits,
        backbone_ms=backbone_ms,
        foot_head_ms=foot_head_ms,
        ulcer_head_ms=ulcer_head_ms,
    )


def guidance_from_foot_ratio(
    foot_area_ratio: float,
    min_foot_ratio: float,
    max_foot_ratio: float,
) -> tuple[bool, str]:
    if foot_area_ratio < min_foot_ratio:
        return False, "발이 충분히 보이지 않습니다. 더 가까이 또는 발 전체가 보이게 촬영하세요."
    if foot_area_ratio > max_foot_ratio:
        return False, "발이 너무 크게 찍혔습니다. 조금 멀리서 촬영하세요."
    return True, "촬영 거리가 적절합니다."


def foot_center_from_mask(mask: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    height, width = mask.shape
    center_x = float(xs.mean() / max(width - 1, 1))
    center_y = float(ys.mean() / max(height - 1, 1))
    return center_x, center_y


def guidance_from_foot_center(
    center_x: float | None,
    center_y: float | None,
    tolerance: float,
) -> tuple[bool, str]:
    if center_x is None or center_y is None:
        return False, "발 위치를 확인할 수 없습니다. 발이 화면 중앙에 오도록 촬영하세요."

    offset_x = center_x - 0.5
    offset_y = center_y - 0.5
    if abs(offset_x) <= tolerance and abs(offset_y) <= tolerance:
        return True, "발이 화면 중앙에 잘 위치해 있습니다."

    directions = []
    if offset_x < -tolerance:
        directions.append("오른쪽")
    elif offset_x > tolerance:
        directions.append("왼쪽")
    if offset_y < -tolerance:
        directions.append("아래쪽")
    elif offset_y > tolerance:
        directions.append("위쪽")

    direction_text = " 및 ".join(directions)
    return False, f"발이 화면 중앙에서 벗어났습니다. 발이 가운데 오도록 {direction_text}으로 조정하세요."


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = mask_image.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def render_overlay(
    image: Image.Image,
    foot_mask: np.ndarray,
    ulcer_mask: np.ndarray,
    alpha: float,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    foot_bool = foot_mask.astype(bool)
    ulcer_bool = ulcer_mask.astype(bool)

    foot_color = np.asarray([0, 128, 255], dtype=np.float32)
    ulcer_color = np.asarray([255, 0, 0], dtype=np.float32)
    overlay[foot_bool] = overlay[foot_bool] * (1.0 - alpha) + foot_color * alpha
    overlay[ulcer_bool] = overlay[ulcer_bool] * (1.0 - alpha) + ulcer_color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


@torch.inference_mode()
def run_gated_segmentation(
    model: MultiTaskSegModel,
    image: Image.Image,
    config: SegmentationConfig,
    device: torch.device,
    *,
    output_size: tuple[int, int],
    autocast_context: Callable[[], Any] | None = None,
) -> GatedSegmentationResult:
    preprocess_start = perf_counter()
    input_tensor = preprocess_image(image, config.image_size, device)
    synchronize_if_needed(device)
    preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

    staged = forward_segmentation_staged(
        model,
        input_tensor,
        device,
        run_ulcer=False,
        autocast_context=autocast_context,
    )
    model_ms = staged.backbone_ms + staged.foot_head_ms

    postprocess_start = perf_counter()
    foot_prob = torch.sigmoid(staged.foot_logits)[0, 0].detach().float().cpu().numpy()
    foot_mask_small = foot_prob > config.foot_threshold
    foot_area_ratio = float(foot_mask_small.mean())
    foot_detected, capture_guidance = guidance_from_foot_ratio(
        foot_area_ratio,
        config.min_foot_ratio,
        config.max_foot_ratio,
    )
    foot_center_x, foot_center_y = foot_center_from_mask(foot_mask_small)
    foot_centered, center_guidance = guidance_from_foot_center(
        foot_center_x,
        foot_center_y,
        config.center_tolerance,
    )
    if foot_detected:
        capture_guidance = "촬영 거리가 적절합니다." if foot_centered else center_guidance

    ulcer_enabled = foot_detected and foot_centered
    ulcer_head_ms = 0.0
    ulcer_mask_small: np.ndarray | None = None
    if ulcer_enabled:
        model_size = tuple(int(value) for value in input_tensor.shape[-2:])
        context_factory = autocast_context or (lambda: nullcontext())
        with context_factory():
            ulcer_start = perf_counter()
            ulcer_logits = model.predict_ulcer_logits(staged.features, model_size)
            synchronize_if_needed(device)
            ulcer_head_ms = (perf_counter() - ulcer_start) * 1000.0
        ulcer_prob = torch.sigmoid(ulcer_logits)[0, 0].detach().float().cpu().numpy()
        ulcer_mask_small = ulcer_prob > config.ulcer_threshold
        ulcer_mask = resize_mask(ulcer_mask_small, output_size)
        ulcer_area_ratio = float(ulcer_mask_small.mean())
    else:
        ulcer_mask = np.zeros(output_size[::-1], dtype=np.uint8)
        ulcer_area_ratio = 0.0

    foot_mask = resize_mask(foot_mask_small, output_size)
    postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
    ulcer_detected = bool(ulcer_enabled and ulcer_area_ratio >= config.min_ulcer_ratio)

    return GatedSegmentationResult(
        foot_mask=foot_mask,
        ulcer_mask=ulcer_mask,
        foot_mask_small=foot_mask_small,
        ulcer_mask_small=ulcer_mask_small,
        foot_detected=foot_detected,
        foot_area_ratio=foot_area_ratio,
        foot_centered=foot_centered,
        foot_center_x=foot_center_x,
        foot_center_y=foot_center_y,
        capture_guidance=capture_guidance,
        ulcer_enabled=ulcer_enabled,
        ulcer_detected=ulcer_detected,
        ulcer_area_ratio=ulcer_area_ratio,
        preprocess_ms=preprocess_ms,
        backbone_ms=staged.backbone_ms,
        foot_head_ms=staged.foot_head_ms,
        model_ms=model_ms,
        ulcer_head_ms=ulcer_head_ms,
        postprocess_ms=postprocess_ms,
    )
