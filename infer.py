from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.diabetic_foot_dataset import IMAGENET_MEAN, IMAGENET_STD
from infer_classification import ClassificationBundle, classify_image, load_classification_bundle
from models import DINOv3Backbone, MultiTaskSegModel
from paths import DEFAULT_CLASSIFICATION_CHECKPOINT
from paths import INFERENCE_OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_HF_MODEL_DIR as DEFAULT_DINOV3_HF_MODEL_DIR
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class InferenceResult:
    image_path: str
    checkpoint_path: str
    foot_detected: bool
    foot_area_ratio: float
    foot_centered: bool
    foot_center_x: float | None
    foot_center_y: float | None
    foot_center_offset_x: float | None
    foot_center_offset_y: float | None
    capture_guidance: str
    ulcer_enabled: bool
    ulcer_detected: bool
    ulcer_area_ratio: float
    foot_threshold: float
    ulcer_threshold: float
    min_foot_ratio: float
    max_foot_ratio: float
    center_tolerance: float
    min_ulcer_ratio: float
    preprocess_ms: float
    model_ms: float
    ulcer_head_ms: float
    postprocess_ms: float
    save_ms: float
    total_ms: float
    fps: float
    foot_mask_path: str
    ulcer_mask_path: str
    overlay_path: str
    classification_enabled: bool
    classification_predicted_class: str | None
    classification_confidence: float | None
    classification_top_k: tuple[dict[str, float | str], ...]
    classification_ms: float
    classification_checkpoint_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gated DFU foot/ulcer segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--foot-threshold", type=float, default=0.5)
    parser.add_argument("--ulcer-threshold", type=float, default=0.5)
    parser.add_argument("--min-foot-ratio", type=float, default=0.01)
    parser.add_argument("--max-foot-ratio", type=float, default=0.8)
    parser.add_argument("--center-tolerance", type=float, default=0.25)
    parser.add_argument("--min-ulcer-ratio", type=float, default=0.001)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument(
        "--classification-checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CHECKPOINT,
        help="DFU classification checkpoint (.pt). Use --no-classification to skip.",
    )
    parser.add_argument(
        "--classification-model-dir",
        type=Path,
        default=DEFAULT_DINOV3_HF_MODEL_DIR,
        help="Local Hugging Face DINOv3 model directory for classification.",
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Disable DFU classification even when a checkpoint is configured.",
    )
    parser.add_argument("--classification-top-k", type=int, default=3)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(args: argparse.Namespace, device: torch.device) -> MultiTaskSegModel:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=True,
    )
    model = MultiTaskSegModel(backbone=backbone).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Input image path not found: {path}")

    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def preprocess_image(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


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


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


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
def predict_image(
    model: MultiTaskSegModel,
    image_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    classification_bundle: ClassificationBundle | None = None,
) -> InferenceResult:
    total_start = perf_counter()

    preprocess_start = perf_counter()
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    input_tensor = preprocess_image(image, args.image_size, device)
    synchronize_if_needed(device)
    preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

    model_start = perf_counter()
    outputs = model(input_tensor)
    foot_logits = outputs["foot"]
    ulcer_logits = outputs["ulcer"]
    synchronize_if_needed(device)
    model_ms = (perf_counter() - model_start) * 1000.0

    postprocess_start = perf_counter()
    foot_prob = torch.sigmoid(foot_logits)[0, 0].detach().cpu().numpy()
    foot_mask_small = foot_prob > args.foot_threshold
    foot_area_ratio = float(foot_mask_small.mean())
    foot_detected, capture_guidance = guidance_from_foot_ratio(
        foot_area_ratio=foot_area_ratio,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
    )
    foot_center_x, foot_center_y = foot_center_from_mask(foot_mask_small)
    foot_centered, center_guidance = guidance_from_foot_center(
        center_x=foot_center_x,
        center_y=foot_center_y,
        tolerance=args.center_tolerance,
    )
    if foot_detected:
        capture_guidance = (
            "촬영 거리가 적절합니다."
            if foot_centered
            else center_guidance
        )

    ulcer_enabled = foot_detected and foot_centered
    ulcer_head_start = perf_counter()
    if ulcer_enabled:
        ulcer_prob = torch.sigmoid(ulcer_logits)[0, 0].detach().cpu().numpy()
        ulcer_mask_small = ulcer_prob > args.ulcer_threshold
        ulcer_mask = resize_mask(ulcer_mask_small, image.size)
        ulcer_area_ratio = float(ulcer_mask.mean())
    else:
        ulcer_mask = np.zeros(image.size[::-1], dtype=np.uint8)
        ulcer_area_ratio = 0.0
    ulcer_head_ms = (perf_counter() - ulcer_head_start) * 1000.0
    ulcer_detected = bool(ulcer_enabled and ulcer_area_ratio >= args.min_ulcer_ratio)

    original_size = image.size
    foot_mask = resize_mask(foot_mask_small, original_size)
    postprocess_ms = (perf_counter() - postprocess_start) * 1000.0

    classification_result = classify_image(
        image=image,
        bundle=classification_bundle,
        device=device,
        enabled=foot_detected and classification_bundle is not None,
        top_k=args.classification_top_k,
    )
    classification_top_k = tuple(
        {"class_name": score.class_name, "probability": score.probability}
        for score in classification_result.top_k
    )

    save_start = perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    foot_mask_path = output_dir / f"{stem}_foot_mask.png"
    ulcer_mask_path = output_dir / f"{stem}_ulcer_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    json_path = output_dir / f"{stem}.json"

    save_mask(foot_mask, foot_mask_path)
    save_mask(ulcer_mask, ulcer_mask_path)
    render_overlay(image, foot_mask, ulcer_mask, args.overlay_alpha).save(overlay_path)
    save_ms = (perf_counter() - save_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    result = InferenceResult(
        image_path=str(image_path),
        checkpoint_path=str(args.checkpoint),
        foot_detected=foot_detected,
        foot_area_ratio=foot_area_ratio,
        foot_centered=foot_centered,
        foot_center_x=round(foot_center_x, 4) if foot_center_x is not None else None,
        foot_center_y=round(foot_center_y, 4) if foot_center_y is not None else None,
        foot_center_offset_x=round(foot_center_x - 0.5, 4) if foot_center_x is not None else None,
        foot_center_offset_y=round(foot_center_y - 0.5, 4) if foot_center_y is not None else None,
        capture_guidance=capture_guidance,
        ulcer_enabled=ulcer_enabled,
        ulcer_detected=ulcer_detected,
        ulcer_area_ratio=ulcer_area_ratio,
        foot_threshold=args.foot_threshold,
        ulcer_threshold=args.ulcer_threshold,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
        center_tolerance=args.center_tolerance,
        min_ulcer_ratio=args.min_ulcer_ratio,
        preprocess_ms=round(preprocess_ms, 2),
        model_ms=round(model_ms, 2),
        ulcer_head_ms=round(ulcer_head_ms, 2),
        postprocess_ms=round(postprocess_ms, 2),
        save_ms=round(save_ms, 2),
        total_ms=round(total_ms, 2),
        fps=round(fps, 2),
        foot_mask_path=str(foot_mask_path),
        ulcer_mask_path=str(ulcer_mask_path),
        overlay_path=str(overlay_path),
        classification_enabled=classification_result.enabled,
        classification_predicted_class=classification_result.predicted_class,
        classification_confidence=(
            round(classification_result.confidence, 4)
            if classification_result.confidence is not None
            else None
        ),
        classification_top_k=classification_top_k,
        classification_ms=classification_result.classification_ms,
        classification_checkpoint_path=classification_result.checkpoint_path,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args, device)
    classification_bundle = None
    if not args.no_classification:
        classification_bundle = load_classification_bundle(
            args.classification_checkpoint,
            device,
            model_dir=args.classification_model_dir,
        )
    image_paths = list(iter_images(args.image))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.image}")

    for image_path in image_paths:
        result = predict_image(model, image_path, args, device, classification_bundle)
        classification_text = (
            f"class={result.classification_predicted_class} "
            f"class_conf={result.classification_confidence:.4f} "
            f"classification_ms={result.classification_ms:.2f} "
            if result.classification_enabled and result.classification_confidence is not None
            else "class=skipped "
        )
        print(
            f"{image_path}: foot={result.foot_detected} "
            f"foot_ratio={result.foot_area_ratio:.4f} "
            f"foot_centered={result.foot_centered} "
            f"ulcer_enabled={result.ulcer_enabled} "
            f"ulcer={result.ulcer_detected} "
            f"ulcer_ratio={result.ulcer_area_ratio:.4f} "
            f"{classification_text}"
            f"model_ms={result.model_ms:.2f} "
            f"ulcer_head_ms={result.ulcer_head_ms:.2f} "
            f"total_ms={result.total_ms:.2f} "
            f"fps={result.fps:.2f} "
            f"guidance={result.capture_guidance}"
        )


if __name__ == "__main__":
    main()
