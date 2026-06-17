from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.diabetic_foot_dataset import IMAGENET_MEAN, IMAGENET_STD
from models import SingleTaskSegModel
from utils.runtime import synchronize_if_needed


@dataclass(frozen=True)
class SegmentationConfig:
    image_size: int
    foot_threshold: float = 0.5
    ulcer_threshold: float = 0.5
    guide_enabled: bool = True
    min_foot_ratio: float = 0.08
    max_foot_ratio: float = 0.5
    center_tolerance: float = 0.25
    min_ulcer_ratio: float = 0.001
    ulcer_feature_crop: bool = True
    ulcer_crop_margin: float = 0.15


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
    features: torch.Tensor = field(repr=False, compare=False)
    foot_mask: np.ndarray
    ulcer_mask: np.ndarray
    foot_mask_small: np.ndarray
    ulcer_mask_small: np.ndarray | None
    ulcer_crop_bbox: tuple[int, int, int, int] | None
    foot_detected: bool
    foot_area_ratio: float
    foot_centered: bool
    foot_center_x: float | None
    foot_center_y: float | None
    capture_guidance: str | None
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
    model: SingleTaskSegModel,
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


def bbox_from_mask(
    mask: np.ndarray,
    margin_ratio: float,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    height, width = mask.shape
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max()) + 1
    y_max = int(ys.max()) + 1
    box_width = max(x_max - x_min, 1)
    box_height = max(y_max - y_min, 1)
    margin_x = int(round(box_width * max(margin_ratio, 0.0)))
    margin_y = int(round(box_height * max(margin_ratio, 0.0)))

    return (
        max(0, x_min - margin_x),
        max(0, y_min - margin_y),
        min(width, x_max + margin_x),
        min(height, y_max + margin_y),
    )


def scale_bbox(
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x_min, y_min, x_max, y_max = bbox
    scaled_x_min = int(np.floor(x_min * target_width / source_width))
    scaled_y_min = int(np.floor(y_min * target_height / source_height))
    scaled_x_max = int(np.ceil(x_max * target_width / source_width))
    scaled_y_max = int(np.ceil(y_max * target_height / source_height))
    return (
        max(0, min(scaled_x_min, target_width - 1)),
        max(0, min(scaled_y_min, target_height - 1)),
        max(1, min(scaled_x_max, target_width)),
        max(1, min(scaled_y_max, target_height)),
    )


def resize_bbox(
    bbox: tuple[int, int, int, int] | None,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    return scale_bbox(bbox, source_size=source_size, target_size=target_size)


def predict_ulcer_logits_for_foot_crop(
    model: SingleTaskSegModel,
    features: torch.Tensor,
    foot_mask: np.ndarray,
    model_size: tuple[int, int],
    margin_ratio: float,
) -> torch.Tensor:
    model_height, model_width = model_size
    image_bbox = bbox_from_mask(foot_mask, margin_ratio)
    if image_bbox is None:
        return model.predict_ulcer_logits(features, model_size)

    feature_height, feature_width = features.shape[-2:]
    feature_bbox = scale_bbox(
        image_bbox,
        source_size=(model_width, model_height),
        target_size=(feature_width, feature_height),
    )
    feature_x_min, feature_y_min, feature_x_max, feature_y_max = feature_bbox
    if feature_x_max <= feature_x_min or feature_y_max <= feature_y_min:
        return model.predict_ulcer_logits(features, model_size)

    image_x_min, image_y_min, image_x_max, image_y_max = image_bbox
    crop_features = features[..., feature_y_min:feature_y_max, feature_x_min:feature_x_max]
    crop_size = (image_y_max - image_y_min, image_x_max - image_x_min)
    crop_logits = model.predict_ulcer_logits(crop_features, crop_size)

    full_logits = torch.full(
        (features.shape[0], 1, model_height, model_width),
        -20.0,
        dtype=crop_logits.dtype,
        device=crop_logits.device,
    )
    full_logits[..., image_y_min:image_y_max, image_x_min:image_x_max] = crop_logits
    return full_logits


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
    ulcer_crop_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    foot_bool = foot_mask.astype(bool)
    ulcer_bool = ulcer_mask.astype(bool)

    foot_color = np.asarray([0, 128, 255], dtype=np.float32)
    ulcer_color = np.asarray([255, 0, 0], dtype=np.float32)
    overlay[foot_bool] = overlay[foot_bool] * (1.0 - alpha) + foot_color * alpha
    overlay[ulcer_bool] = overlay[ulcer_bool] * (1.0 - alpha) + ulcer_color * alpha
    rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    if ulcer_crop_bbox is not None:
        draw = ImageDraw.Draw(rendered)
        x_min, y_min, x_max, y_max = ulcer_crop_bbox
        for inset in range(3):
            draw.rectangle(
                (x_min + inset, y_min + inset, max(x_min, x_max - 1 - inset), max(y_min, y_max - 1 - inset)),
                outline=(255, 220, 0),
            )
    return rendered


@torch.inference_mode()
def run_gated_segmentation(
    model: SingleTaskSegModel,
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
    foot_present = bool(foot_mask_small.any())
    foot_center_x, foot_center_y = foot_center_from_mask(foot_mask_small)

    if config.guide_enabled:
        foot_detected, capture_guidance = guidance_from_foot_ratio(
            foot_area_ratio,
            config.min_foot_ratio,
            config.max_foot_ratio,
        )
        foot_centered, center_guidance = guidance_from_foot_center(
            foot_center_x,
            foot_center_y,
            config.center_tolerance,
        )
        if foot_detected:
            capture_guidance = "촬영 거리가 적절합니다." if foot_centered else center_guidance
        ulcer_enabled = foot_detected and foot_centered
    else:
        foot_detected = foot_present
        foot_centered = foot_present
        capture_guidance = None
        ulcer_enabled = foot_present

    ulcer_head_ms = 0.0
    ulcer_mask_small: np.ndarray | None = None
    model_size = tuple(int(value) for value in input_tensor.shape[-2:])
    model_height, model_width = model_size
    ulcer_crop_bbox_small = (
        bbox_from_mask(foot_mask_small, config.ulcer_crop_margin)
        if ulcer_enabled and config.ulcer_feature_crop
        else None
    )
    if ulcer_enabled:
        context_factory = autocast_context or (lambda: nullcontext())
        with context_factory():
            ulcer_start = perf_counter()
            if config.ulcer_feature_crop:
                ulcer_logits = predict_ulcer_logits_for_foot_crop(
                    model,
                    staged.features,
                    foot_mask_small,
                    model_size,
                    config.ulcer_crop_margin,
                )
            else:
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
    ulcer_crop_bbox = resize_bbox(
        ulcer_crop_bbox_small,
        source_size=(model_width, model_height),
        target_size=output_size,
    )
    postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
    ulcer_detected = bool(ulcer_enabled and ulcer_area_ratio >= config.min_ulcer_ratio)

    return GatedSegmentationResult(
        features=staged.features,
        foot_mask=foot_mask,
        ulcer_mask=ulcer_mask,
        foot_mask_small=foot_mask_small,
        ulcer_mask_small=ulcer_mask_small,
        ulcer_crop_bbox=ulcer_crop_bbox,
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
