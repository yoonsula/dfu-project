from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paths import DEFAULT_BODY_ROOT
from paths import DEFAULT_DFU_FOOT_ROOT
from paths import DEFAULT_FOOT_ROOT
from paths import DEFAULT_HUMANBODY_ROOT
from paths import DEFAULT_WOUND_ROOT
from paths import DEFAULT_WOUND_IMAGE_ROOT

BODY_FOOT_CATEGORY_IDS = {1}
HUMANBODY_FOOT_CATEGORY_IDS = {5, 10}


@dataclass(frozen=True)
class CocoSource:
    name: str
    root: Path
    category_ids: set[int] | None = None
    positive_profile: str = "natural"
    negative_profile: str = "none"
    missing_ok: bool = False


@dataclass(frozen=True)
class FusegSource:
    name: str
    root: Path


@dataclass(frozen=True)
class WoundImageSource:
    name: str
    root: Path


def default_foot_roots() -> tuple[Path, ...]:
    roots = [DEFAULT_FOOT_ROOT]
    if (DEFAULT_DFU_FOOT_ROOT / "_annotations.coco.json").exists():
        roots.append(DEFAULT_DFU_FOOT_ROOT)
    return tuple(roots)


def foot_primary_sources(
    foot_roots: tuple[Path, ...],
    *,
    positive_profile: str,
) -> list[CocoSource]:
    return [
        CocoSource(
            name=f"foot:{root.name}",
            root=root,
            positive_profile=positive_profile,
            negative_profile="none",
            missing_ok=True,
        )
        for root in foot_roots
    ]


def foot_extra_coco_sources(
    *,
    body_root: Path | None,
    humanbody_root: Path | None,
) -> list[CocoSource]:
    sources: list[CocoSource] = []
    if body_root is not None:
        sources.append(
            CocoSource(
                name="roboflow-body",
                root=body_root,
                category_ids=BODY_FOOT_CATEGORY_IDS,
                positive_profile="natural",
                negative_profile="negative_fullbody",
            )
        )
    if humanbody_root is not None:
        sources.append(
            CocoSource(
                name="roboflow-humanbody",
                root=humanbody_root,
                category_ids=HUMANBODY_FOOT_CATEGORY_IDS,
                positive_profile="natural",
                negative_profile="negative_fullbody",
            )
        )
    return sources


def wound_sources(
    *,
    wound_root: Path = DEFAULT_WOUND_ROOT,
    wound_image_root: Path | None = DEFAULT_WOUND_IMAGE_ROOT,
) -> tuple[FusegSource, WoundImageSource | None]:
    fuseg = FusegSource(name="fuseg", root=wound_root)
    wound = WoundImageSource(name="wound-image", root=wound_image_root) if wound_image_root else None
    return fuseg, wound


__all__ = [
    "BODY_FOOT_CATEGORY_IDS",
    "HUMANBODY_FOOT_CATEGORY_IDS",
    "CocoSource",
    "FusegSource",
    "WoundImageSource",
    "default_foot_roots",
    "foot_primary_sources",
    "foot_extra_coco_sources",
    "wound_sources",
]
