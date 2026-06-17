"""Check that bundled models and assets are present before inference."""

from __future__ import annotations

import sys
from pathlib import Path

from paths import (
    DEFAULT_CLASSIFICATION_CHECKPOINT,
    DINOV3_CHECKPOINT,
    DINOV3_HF_MODEL_DIR,
    DINOV3_REPO,
)


def main() -> int:
    required = [
        ("Classification checkpoint", DEFAULT_CLASSIFICATION_CHECKPOINT),
        ("DINOv3 repo (segmentation)", DINOV3_REPO),
        ("DINOv3 backbone weights", DINOV3_CHECKPOINT),
        ("DINOv3 HF model (classification)", DINOV3_HF_MODEL_DIR),
    ]
    missing: list[tuple[str, Path]] = []
    for label, path in required:
        if not path.exists():
            missing.append((label, path))

    if missing:
        print("Missing required assets:")
        for label, path in missing:
            print(f"  - {label}: {path}")
        return 1

    print("All required assets are present.")
    for label, path in required:
        print(f"  OK  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
