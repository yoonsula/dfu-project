"""Check that the local Hugging Face DINOv3 backbone snapshot can be loaded."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import DINOV3_MODEL_PATH


def main() -> int:
    print(f"Checking DINOv3 backbone: {DINOV3_MODEL_PATH}")
    if not DINOV3_MODEL_PATH.exists():
        print(
            "Missing local model directory. Expected a Hugging Face snapshot with "
            "config.json and model weights."
        )
        return 1

    try:
        from models import DINOv3Backbone

        backbone = DINOv3Backbone(model_path=DINOV3_MODEL_PATH, freeze=True)
        backbone.eval()
    except Exception as exc:
        print(f"Failed to load backbone: {exc}")
        return 1

    print("DINOv3 backbone loaded successfully.")
    print(f"  model_path: {backbone.model_path}")
    print(f"  feature_dim: {backbone.feature_dim}")
    print(f"  local_files_only: {backbone.local_files_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
