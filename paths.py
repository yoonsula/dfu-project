"""Central path configuration. Paths default to project-relative locations (see .env.example)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Allow `python ../dfu-project/infer.py` or running from any cwd.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(env_key: str, default: Path, fallback_env_key: str | None = None) -> Path:
    """Resolve env path; relative values are interpreted from PROJECT_ROOT."""
    raw = os.environ.get(env_key)
    if raw is None and fallback_env_key is not None:
        raw = os.environ.get(fallback_env_key)
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# --- Inference assets (local Hugging Face backbone snapshot + head checkpoints) ---

DINOV3_MODEL_PATH = _resolve_path(
    "DINOV3_MODEL_PATH",
    ASSETS_DIR / "dinov3-hf",
    fallback_env_key="DINOV3_MODEL_ID",
)

CHECKPOINT_DIR = _resolve_path("DFU_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")
TRAIN_OUTPUT_DIR = _resolve_path(
    "DFU_TRAIN_OUTPUT_DIR",
    CHECKPOINT_DIR,
)

INFERENCE_OUTPUT_DIR = _resolve_path(
    "DFU_INFERENCE_OUTPUT_DIR",
    PROJECT_ROOT / "output" / "inference",
)

# --- Training data (not bundled; override via .env when retraining) ---

DATA_ROOT = _resolve_path("DFU_DATA_ROOT", PROJECT_ROOT / ".." / ".." / "03_데이터")
DEFAULT_FOOT_ROOT = _resolve_path("DFU_FOOT_ROOT", DATA_ROOT / "roboflow-foot")
DEFAULT_DFU_FOOT_ROOT = _resolve_path(
    "DFU_DFU_FOOT_ROOT",
    DATA_ROOT / "dfu-foot-sam3-filtered" / "train",
)
DEFAULT_BODY_ROOT = _resolve_path("DFU_BODY_ROOT", DATA_ROOT / "roboflow-body")
DEFAULT_HUMANBODY_ROOT = _resolve_path("DFU_HUMANBODY_ROOT", DATA_ROOT / "roboflow-humanbody")
DEFAULT_WOUND_ROOT = _resolve_path(
    "DFU_WOUND_ROOT",
    DATA_ROOT / "wound-segmentation" / "data" / "Foot Ulcer Segmentation Challenge",
    fallback_env_key="DFU_ULCER_ROOT",
)
DEFAULT_WOUND_IMAGE_ROOT = _resolve_path(
    "DFU_WOUND_IMAGE_ROOT",
    DATA_ROOT / "Wound Image Dataset",
)
DEFAULT_DFU_CLASSIFICATION_SOURCE_ROOT = _resolve_path(
    "DFU_CLASSIFICATION_SOURCE_ROOT",
    DATA_ROOT / "DFU Dataset",
)
DEFAULT_DFU_PARTA_ROOT = _resolve_path(
    "DFU_PARTA_ROOT",
    DATA_ROOT / "dfu_partA_20260617",
)
DEFAULT_DFU_CLASSIFICATION_DATA_ROOT = _resolve_path(
    "DFU_CLASSIFICATION_DATA_ROOT",
    DATA_ROOT / "dfu_classification_data",
)
