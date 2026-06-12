from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from paths import DEFAULT_FOOT_ROOT
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scale-augmented Roboflow foot COCO dataset.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_FOOT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--scale-min", type=float, default=1.5)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-originals", action="store_true")
    return parser.parse_args()


def load_coco(input_root: Path) -> dict[str, Any]:
    annotation_path = input_root / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def sample_transform(
    width: int,
    height: int,
    scale_min: float,
    scale_max: float,
    hflip_prob: float,
) -> dict[str, float | bool | int]:
    scale = random.uniform(scale_min, scale_max)
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))

    if scaled_width >= width:
        offset_x = -random.randint(0, scaled_width - width)
    else:
        offset_x = random.randint(0, width - scaled_width)

    if scaled_height >= height:
        offset_y = -random.randint(0, scaled_height - height)
    else:
        offset_y = random.randint(0, height - scaled_height)

    return {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "hflip": random.random() < hflip_prob,
    }


def transform_image(image: Image.Image, transform: dict[str, float | bool | int]) -> Image.Image:
    width, height = image.size
    scaled_width = int(transform["scaled_width"])
    scaled_height = int(transform["scaled_height"])
    offset_x = int(transform["offset_x"])
    offset_y = int(transform["offset_y"])
    hflip = bool(transform["hflip"])

    scaled = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(scaled, (offset_x, offset_y))
    if hflip:
        canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return canvas


def transform_point(
    x: float,
    y: float,
    width: int,
    transform: dict[str, float | bool | int],
) -> tuple[float, float]:
    scale = float(transform["scale"])
    offset_x = int(transform["offset_x"])
    offset_y = int(transform["offset_y"])
    hflip = bool(transform["hflip"])

    x_t = x * scale + offset_x
    y_t = y * scale + offset_y
    if hflip:
        x_t = width - 1 - x_t
    return x_t, y_t


def clip_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)


def transform_polygon(
    polygon: list[float],
    width: int,
    height: int,
    transform: dict[str, float | bool | int],
) -> list[float]:
    points: list[tuple[float, float]] = []
    for index in range(0, len(polygon), 2):
        x_t, y_t = transform_point(polygon[index], polygon[index + 1], width, transform)
        points.append(clip_point(x_t, y_t, width, height))
    if bool(transform["hflip"]):
        points.reverse()
    return [coord for point in points for coord in point]


def polygon_area(polygon: list[float]) -> float:
    if len(polygon) < 6:
        return 0.0
    points = [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon), 2)]
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_bbox(polygons: list[list[float]]) -> list[float]:
    xs = [polygon[index] for polygon in polygons for index in range(0, len(polygon), 2)]
    ys = [polygon[index] for polygon in polygons for index in range(1, len(polygon), 2)]
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def transform_annotation(
    annotation: dict[str, Any],
    new_id: int,
    new_image_id: int,
    width: int,
    height: int,
    transform: dict[str, float | bool | int],
) -> dict[str, Any] | None:
    segmentation = annotation.get("segmentation", [])
    if not isinstance(segmentation, list):
        return None

    polygons = [
        transform_polygon(polygon, width, height, transform)
        for polygon in segmentation
        if isinstance(polygon, list) and len(polygon) >= 6
    ]
    polygons = [polygon for polygon in polygons if polygon_area(polygon) > 1.0]
    if not polygons:
        return None

    new_annotation = copy.deepcopy(annotation)
    new_annotation["id"] = new_id
    new_annotation["image_id"] = new_image_id
    new_annotation["segmentation"] = polygons
    new_annotation["bbox"] = polygon_bbox(polygons)
    new_annotation["area"] = sum(polygon_area(polygon) for polygon in polygons)
    return new_annotation


def copy_originals(
    coco: dict[str, Any],
    input_root: Path,
    output_root: Path,
    output: dict[str, Any],
) -> tuple[int, int]:
    next_image_id = 0
    next_annotation_id = 0
    image_id_map: dict[int, int] = {}

    for image_info in coco.get("images", []):
        src = input_root / image_info["file_name"]
        if not src.exists():
            continue
        dst = output_root / image_info["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as image:
            image.convert("RGB").save(dst)

        copied = copy.deepcopy(image_info)
        image_id_map[int(image_info["id"])] = next_image_id
        copied["id"] = next_image_id
        output["images"].append(copied)
        next_image_id += 1

    for annotation in coco.get("annotations", []):
        old_image_id = int(annotation["image_id"])
        if old_image_id not in image_id_map:
            continue
        copied = copy.deepcopy(annotation)
        copied["id"] = next_annotation_id
        copied["image_id"] = image_id_map[old_image_id]
        output["annotations"].append(copied)
        next_annotation_id += 1

    return next_image_id, next_annotation_id


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.input_root)
    grouped = annotations_by_image(coco)
    output = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco.get("categories", [])),
        "images": [],
        "annotations": [],
    }

    next_image_id = 0
    next_annotation_id = 0
    if args.include_originals:
        next_image_id, next_annotation_id = copy_originals(coco, args.input_root, args.output_root, output)

    exported_source_count = 0
    for image_info in coco.get("images", []):
        if args.limit is not None and exported_source_count >= args.limit:
            break
        src = args.input_root / image_info["file_name"]
        if not src.exists() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        with Image.open(src) as raw_image:
            image = raw_image.convert("RGB")
        width, height = image.size

        for copy_index in range(args.copies):
            transform = sample_transform(width, height, args.scale_min, args.scale_max, args.hflip_prob)
            aug_name = f"{Path(image_info['file_name']).stem}_aug{copy_index}{src.suffix.lower()}"
            transform_image(image, transform).save(args.output_root / aug_name)

            output["images"].append(
                {
                    **copy.deepcopy(image_info),
                    "id": next_image_id,
                    "file_name": aug_name,
                    "width": width,
                    "height": height,
                }
            )

            for annotation in grouped.get(int(image_info["id"]), []):
                transformed = transform_annotation(
                    annotation,
                    new_id=next_annotation_id,
                    new_image_id=next_image_id,
                    width=width,
                    height=height,
                    transform=transform,
                )
                if transformed is None:
                    continue
                output["annotations"].append(transformed)
                next_annotation_id += 1
            next_image_id += 1
        exported_source_count += 1

    with (args.output_root / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False)

    print(
        f"Exported {len(output['images'])} images and {len(output['annotations'])} annotations "
        f"to {args.output_root}"
    )


if __name__ == "__main__":
    main()
