from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference.pipeline import SegmentationConfig
from inference.pipeline import run_gated_segmentation
from inference.pipeline import render_overlay
from infer_classification import ClassificationBundle, classify_image, load_classification_bundle
from infer_classification import ClassificationResult, ClassScore
from models import DINOv3Backbone, DFUFeatureClassifierHead, SingleTaskSegModel
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
    capture_guidance: str | None
    guide_enabled: bool
    wound_enabled: bool
    wound_detected: bool
    wound_area_ratio: float
    wound_crop_bbox: tuple[int, int, int, int] | None
    foot_threshold: float
    wound_threshold: float
    min_foot_ratio: float
    max_foot_ratio: float
    center_tolerance: float
    min_wound_ratio: float
    wound_feature_crop: bool
    wound_crop_margin: float
    preprocess_ms: float
    backbone_ms: float
    foot_head_ms: float
    model_ms: float
    wound_head_ms: float
    postprocess_ms: float
    save_ms: float
    total_ms: float
    fps: float
    foot_mask_path: str
    wound_mask_path: str
    overlay_path: str
    classification_enabled: bool
    classification_predicted_class: str | None
    classification_confidence: float | None
    classification_top_k: tuple[dict[str, float | str], ...]
    classification_ms: float
    classification_checkpoint_path: str | None


@dataclass(frozen=True)
class DFUHeadBundle:
    head: DFUFeatureClassifierHead
    id2label: dict[int, str]
    classes: tuple[str, ...]
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gated DFU foot/wound segmentation inference.")
    parser.add_argument(
        "--foot-head-checkpoint",
        type=Path,
        required=True,
        help="Head-only foot segmentation checkpoint trained on shared DINOv3 features.",
    )
    parser.add_argument(
        "--wound-head-checkpoint",
        type=Path,
        required=True,
        help="Head-only wound segmentation checkpoint trained on shared DINOv3 features.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input resolution. Defaults to foot head checkpoint args.image_size, then 384.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--foot-threshold", type=float, default=0.5)
    parser.add_argument("--wound-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Disable capture guidance and do not let guidance gates block wound/classification stages.",
    )
    parser.add_argument("--min-foot-ratio", type=float, default=0.08)
    parser.add_argument("--max-foot-ratio", type=float, default=0.5)
    parser.add_argument("--center-tolerance", type=float, default=0.25)
    parser.add_argument("--min-wound-ratio", type=float, default=0.001)
    parser.add_argument(
        "--wound-crop-margin",
        type=float,
        default=0.15,
        help="Margin ratio around the detected foot bbox when cropping shared features for wound head.",
    )
    parser.add_argument(
        "--no-wound-feature-crop",
        action="store_true",
        help="Run the wound head on the full feature map instead of the detected foot feature crop.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument(
        "--classification-checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CHECKPOINT,
        help="DFU classification checkpoint with its own HuggingFace DINOv3 backbone.",
    )
    parser.add_argument(
        "--dfu-head-checkpoint",
        type=Path,
        default=None,
        help="DFU classification head checkpoint that consumes shared DINOv3 feature maps.",
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


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like torch payload.")
    for key in ("head_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain head_state_dict or state_dict weights.")


def _strip_first_matching_prefix(
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
    state_dict = _strip_first_matching_prefix(_checkpoint_state_dict(checkpoint), prefixes)
    missing_keys, unexpected_keys = head.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        warnings.warn(
            f"{checkpoint_path} loaded with non-strict head key mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )


def load_model(args: argparse.Namespace, device: torch.device) -> SingleTaskSegModel:
    backbone = DINOv3Backbone(
        repo_dir=args.dinov3_repo,
        checkpoint_path=args.dinov3_checkpoint,
        freeze=True,
    )
    model = SingleTaskSegModel(backbone=backbone).to(device)
    if args.foot_head_checkpoint is not None:
        load_segmentation_head(
            model.foot_head,
            args.foot_head_checkpoint,
            prefixes=("foot_head.", "head."),
            device=device,
        )
    if args.wound_head_checkpoint is not None:
        load_segmentation_head(
            model.wound_head,
            args.wound_head_checkpoint,
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


def load_dfu_head_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> DFUHeadBundle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DFU head checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"DFU head checkpoint must be a dict: {checkpoint_path}")

    classes = tuple(checkpoint.get("classes", ("TS6_normal skin", "diabetic wound", "other_injury")))
    id2label_raw = checkpoint.get("id2label", {index: label for index, label in enumerate(classes)})
    id2label = {int(key): str(value) for key, value in id2label_raw.items()}
    feature_dim = int(checkpoint.get("feature_dim", 384))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    dropout = float(checkpoint.get("dropout", 0.2))
    head_type = str(checkpoint.get("head_type", "mlp"))
    head = DFUFeatureClassifierHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        dropout=dropout,
        head_type=head_type,
    ).to(device)
    state_dict = _strip_first_matching_prefix(
        _checkpoint_state_dict(checkpoint),
        ("classification_head.", "dfu_head.", "head."),
    )
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    return DFUHeadBundle(
        head=head,
        id2label=id2label,
        classes=classes,
        checkpoint_path=checkpoint_path,
    )


@torch.inference_mode()
def classify_shared_features(
    features: torch.Tensor,
    bundle: DFUHeadBundle | None,
    *,
    enabled: bool,
    top_k: int,
) -> ClassificationResult:
    checkpoint_path = str(bundle.checkpoint_path) if bundle is not None else None
    if not enabled or bundle is None:
        return ClassificationResult(
            enabled=False,
            predicted_class=None,
            confidence=None,
            top_k=(),
            classification_ms=0.0,
            checkpoint_path=checkpoint_path,
        )

    start = perf_counter()
    logits = bundle.head(features)
    probabilities = F.softmax(logits, dim=-1)[0]
    k = min(top_k, len(bundle.classes))
    top_probs, top_indices = torch.topk(probabilities, k=k)
    scores = tuple(
        ClassScore(
            class_name=bundle.id2label[int(index)],
            probability=float(prob),
        )
        for prob, index in zip(top_probs.cpu(), top_indices.cpu())
    )
    classification_ms = (perf_counter() - start) * 1000.0
    return ClassificationResult(
        enabled=True,
        predicted_class=scores[0].class_name if scores else None,
        confidence=scores[0].probability if scores else None,
        top_k=scores,
        classification_ms=round(classification_ms, 2),
        checkpoint_path=checkpoint_path,
    )


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


@torch.inference_mode()
def predict_image(
    model: SingleTaskSegModel,
    image_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    classification_bundle: ClassificationBundle | None = None,
    dfu_head_bundle: DFUHeadBundle | None = None,
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
            wound_threshold=args.wound_threshold,
            guide_enabled=not args.no_guide,
            min_foot_ratio=args.min_foot_ratio,
            max_foot_ratio=args.max_foot_ratio,
            center_tolerance=args.center_tolerance,
            min_wound_ratio=args.min_wound_ratio,
            wound_feature_crop=not args.no_wound_feature_crop,
            wound_crop_margin=args.wound_crop_margin,
        ),
        device,
        output_size=image.size,
    )

    classification_enabled = segmentation.foot_detected
    if dfu_head_bundle is not None:
        classification_result = classify_shared_features(
            segmentation.features,
            dfu_head_bundle,
            enabled=classification_enabled,
            top_k=args.classification_top_k,
        )
    else:
        classification_result = classify_image(
            image=image,
            bundle=classification_bundle,
            device=device,
            enabled=classification_enabled and classification_bundle is not None,
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
    wound_mask_path = output_dir / f"{stem}_wound_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    json_path = output_dir / f"{stem}.json"

    save_mask(segmentation.foot_mask, foot_mask_path)
    save_mask(segmentation.wound_mask, wound_mask_path)
    render_overlay(
        image,
        segmentation.foot_mask,
        segmentation.wound_mask,
        args.overlay_alpha,
        segmentation.wound_crop_bbox,
    ).save(overlay_path)
    save_ms = (perf_counter() - save_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    result = InferenceResult(
        image_path=str(image_path),
        checkpoint_path=str(args.foot_head_checkpoint),
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
        capture_guidance=None if args.no_guide else segmentation.capture_guidance,
        guide_enabled=not args.no_guide,
        wound_enabled=segmentation.wound_enabled,
        wound_detected=segmentation.wound_detected,
        wound_area_ratio=segmentation.wound_area_ratio,
        wound_crop_bbox=segmentation.wound_crop_bbox,
        foot_threshold=args.foot_threshold,
        wound_threshold=args.wound_threshold,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
        center_tolerance=args.center_tolerance,
        min_wound_ratio=args.min_wound_ratio,
        wound_feature_crop=not args.no_wound_feature_crop,
        wound_crop_margin=args.wound_crop_margin,
        preprocess_ms=round(segmentation.preprocess_ms, 2),
        backbone_ms=round(segmentation.backbone_ms, 2),
        foot_head_ms=round(segmentation.foot_head_ms, 2),
        model_ms=round(segmentation.model_ms, 2),
        wound_head_ms=round(segmentation.wound_head_ms, 2),
        postprocess_ms=round(segmentation.postprocess_ms, 2),
        save_ms=round(save_ms, 2),
        total_ms=round(total_ms, 2),
        fps=round(fps, 2),
        foot_mask_path=str(foot_mask_path),
        wound_mask_path=str(wound_mask_path),
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
    args.image_size = resolve_image_size_from_checkpoint(args.foot_head_checkpoint, args.image_size)
    device = resolve_device(args.device)
    model = load_model(args, device)
    classification_bundle = None
    dfu_head_bundle = None
    if not args.no_classification:
        if args.dfu_head_checkpoint is not None:
            dfu_head_bundle = load_dfu_head_bundle(
                args.dfu_head_checkpoint,
                device,
            )
        else:
            classification_bundle = load_classification_bundle(
                args.classification_checkpoint,
                device,
                model_dir=args.classification_model_dir,
            )
    image_paths = list(iter_images(args.image))
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.image}")

    for image_path in image_paths:
        result = predict_image(
            model,
            image_path,
            args,
            device,
            classification_bundle,
            dfu_head_bundle,
        )
        classification_text = (
            f"class={result.classification_predicted_class} "
            f"class_conf={result.classification_confidence:.4f} "
            f"classification_ms={result.classification_ms:.2f} "
            if result.classification_enabled and result.classification_confidence is not None
            else "class=skipped "
        )
        guidance_text = (
            f"guidance={result.capture_guidance}"
            if result.capture_guidance is not None
            else ""
        )
        print(
            f"{image_path}: foot={result.foot_detected} "
            f"foot_ratio={result.foot_area_ratio:.4f} "
            f"foot_centered={result.foot_centered} "
            f"wound_enabled={result.wound_enabled} "
            f"wound={result.wound_detected} "
            f"wound_ratio={result.wound_area_ratio:.4f} "
            f"{classification_text}"
            f"backbone_ms={result.backbone_ms:.2f} "
            f"foot_head_ms={result.foot_head_ms:.2f} "
            f"model_ms={result.model_ms:.2f} "
            f"wound_head_ms={result.wound_head_ms:.2f} "
            f"total_ms={result.total_ms:.2f} "
            f"fps={result.fps:.2f} "
            f"{guidance_text}"
        )


if __name__ == "__main__":
    main()
