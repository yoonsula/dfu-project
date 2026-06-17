from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from inference.pipeline import SegmentationConfig
from inference.pipeline import run_gated_segmentation
from inference.pipeline import render_overlay
from infer_classification import ClassificationBundle, classify_image, load_classification_bundle
from models import DINOv3Backbone, MultiTaskSegModel
from paths import DEFAULT_CLASSIFICATION_CHECKPOINT
from paths import INFERENCE_OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_HF_MODEL_DIR as DEFAULT_DINOV3_HF_MODEL_DIR
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from utils.image_io import iter_images
from utils.runtime import resolve_device

DEFAULT_IMAGE_SIZE = 384


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
    backbone_ms: float
    foot_head_ms: float
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input resolution. Defaults to checkpoint args.image_size, then 384.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--foot-threshold", type=float, default=0.5)
    parser.add_argument("--ulcer-threshold", type=float, default=0.5)
    parser.add_argument("--min-foot-ratio", type=float, default=0.08)
    parser.add_argument("--max-foot-ratio", type=float, default=0.5)
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
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        warnings.warn(
            "Checkpoint loaded with non-strict key mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )
    model.eval()
    return model


def resolve_image_size_from_checkpoint(
    checkpoint_path: Path,
    requested_image_size: int | None,
) -> int:
    if requested_image_size is not None:
        return int(requested_image_size)
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


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


@torch.inference_mode()
def predict_image(
    model: MultiTaskSegModel,
    image_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    classification_bundle: ClassificationBundle | None = None,
) -> InferenceResult:
    total_start = perf_counter()

    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    segmentation = run_gated_segmentation(
        model,
        image,
        SegmentationConfig(
            image_size=args.image_size,
            foot_threshold=args.foot_threshold,
            ulcer_threshold=args.ulcer_threshold,
            min_foot_ratio=args.min_foot_ratio,
            max_foot_ratio=args.max_foot_ratio,
            center_tolerance=args.center_tolerance,
            min_ulcer_ratio=args.min_ulcer_ratio,
        ),
        device,
        output_size=image.size,
    )

    classification_result = classify_image(
        image=image,
        bundle=classification_bundle,
        device=device,
        enabled=segmentation.foot_detected and classification_bundle is not None,
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

    save_mask(segmentation.foot_mask, foot_mask_path)
    save_mask(segmentation.ulcer_mask, ulcer_mask_path)
    render_overlay(
        image,
        segmentation.foot_mask,
        segmentation.ulcer_mask,
        args.overlay_alpha,
    ).save(overlay_path)
    save_ms = (perf_counter() - save_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    result = InferenceResult(
        image_path=str(image_path),
        checkpoint_path=str(args.checkpoint),
        foot_detected=segmentation.foot_detected,
        foot_area_ratio=segmentation.foot_area_ratio,
        foot_centered=segmentation.foot_centered,
        foot_center_x=round(segmentation.foot_center_x, 4)
        if segmentation.foot_center_x is not None
        else None,
        foot_center_y=round(segmentation.foot_center_y, 4)
        if segmentation.foot_center_y is not None
        else None,
        foot_center_offset_x=round(segmentation.foot_center_x - 0.5, 4)
        if segmentation.foot_center_x is not None
        else None,
        foot_center_offset_y=round(segmentation.foot_center_y - 0.5, 4)
        if segmentation.foot_center_y is not None
        else None,
        capture_guidance=segmentation.capture_guidance,
        ulcer_enabled=segmentation.ulcer_enabled,
        ulcer_detected=segmentation.ulcer_detected,
        ulcer_area_ratio=segmentation.ulcer_area_ratio,
        foot_threshold=args.foot_threshold,
        ulcer_threshold=args.ulcer_threshold,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
        center_tolerance=args.center_tolerance,
        min_ulcer_ratio=args.min_ulcer_ratio,
        preprocess_ms=round(segmentation.preprocess_ms, 2),
        backbone_ms=round(segmentation.backbone_ms, 2),
        foot_head_ms=round(segmentation.foot_head_ms, 2),
        model_ms=round(segmentation.model_ms, 2),
        ulcer_head_ms=round(segmentation.ulcer_head_ms, 2),
        postprocess_ms=round(segmentation.postprocess_ms, 2),
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
    args.image_size = resolve_image_size_from_checkpoint(args.checkpoint, args.image_size)
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
            f"backbone_ms={result.backbone_ms:.2f} "
            f"foot_head_ms={result.foot_head_ms:.2f} "
            f"model_ms={result.model_ms:.2f} "
            f"ulcer_head_ms={result.ulcer_head_ms:.2f} "
            f"total_ms={result.total_ms:.2f} "
            f"fps={result.fps:.2f} "
            f"guidance={result.capture_guidance}"
        )


if __name__ == "__main__":
    main()
