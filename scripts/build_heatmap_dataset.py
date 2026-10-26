"""Save this file in LilyCascade/scripts, then run it in PyCharm.

Build an anther-only heatmap dataset from lily boxes and CVAT tip points.
Source photographs, annotations, and split assignments are never modified.
"""

from pathlib import Path
import hashlib
import html
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


# These are dataset settings, not model training hyperparameters.
IMAGE_SIZE = 256
CROP_MARGIN = 0.10  # Add 10% of box width/height on EACH side.
SIGMA = 2.0  # Gaussian spread, measured in final heatmap pixels.
DATASET_NAME = "anther_crops_v1"


def read_endpoints(annotation_directory):
    """An explicit zero-point entry differs from a missing annotation."""
    entries = {}
    xml_files = sorted(annotation_directory.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in {annotation_directory}")
    for xml_file in xml_files:
        for entry in ET.parse(xml_file).getroot().findall("image"):
            name = entry.get("name")
            if name in entries:
                raise ValueError(f"Duplicate XML image entry: {name}")
            width, height = int(entry.get("width")), int(entry.get("height"))
            points = []
            for shape in entry.findall("points"):
                if shape.get("label") != "anther_tip":
                    raise ValueError(f"Unexpected point label in {name}")
                for pair in shape.get("points").split(";"):
                    x, y = map(float, pair.split(","))
                    if not (math.isfinite(x) and math.isfinite(y)
                            and 0 <= x < width and 0 <= y < height):
                        raise ValueError(f"Invalid point in {name}: {pair}")
                    points.append([x, y])
            entries[name] = {"size": (width, height), "points": points}
    return entries, xml_files


def read_crop_boxes(label_path, width, height):
    """Convert normalized, lily-only YOLO boxes to padded pixel crops."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid YOLO row in {label_path}")
        class_id, cx, cy, bw, bh = values
        if class_id != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1
                                and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Expected lily-only detection labels: {label_path}")
        margin = 0.5 + CROP_MARGIN
        box = [max(0, math.floor((cx - margin * bw) * width)),
               max(0, math.floor((cy - margin * bh) * height)),
               min(width, math.ceil((cx + margin * bw) * width)),
               min(height, math.ceil((cy + margin * bh) * height))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Empty crop in {label_path}")
        boxes.append(box)
    return boxes


def make_heatmap(points):
    """Combine Gaussian peaks with maximum, keeping targets in [0, 1]."""
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    target = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for x, y in points:
        peak = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2))
        target = np.maximum(target, peak.astype(np.float32))
    return target


def prepare_sample(image, box, points):
    """Apply identical crop/resize/padding geometry to image and points."""
    x0, y0, x1, y1 = box
    crop = image.crop(box)
    scale = IMAGE_SIZE / max(crop.size)
    new_width = max(1, round(crop.width * scale))
    new_height = max(1, round(crop.height * scale))
    pad_x = (IMAGE_SIZE - new_width) // 2
    pad_y = (IMAGE_SIZE - new_height) // 2
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114))
    canvas.paste(crop.resize((new_width, new_height), Image.Resampling.BILINEAR),
                 (pad_x, pad_y))
    indices = [i for i, (x, y) in enumerate(points)
               if x0 <= x < x1 and y0 <= y < y1]
    transformed = [[(points[i][0] - x0) * new_width / crop.width + pad_x,
                    (points[i][1] - y0) * new_height / crop.height + pad_y]
                   for i in indices]
    geometry = {"crop_xyxy": box, "resized_wh": [new_width, new_height],
                "padding_xy": [pad_x, pad_y],
                "scale_xy": [new_width / crop.width, new_height / crop.height]}
    return canvas, transformed, indices, geometry


def main(project_root=None, output_root=None):
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    source = project_root / "datasets" / "lily_v1"
    annotation_directory = project_root / "datasets/annotations/anther_endpoints"
    output = Path(output_root or project_root / "datasets" / DATASET_NAME)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new dataset name: {output}")
    entries, xml_files = read_endpoints(annotation_directory)

    # Preflight: verify annotations and coverage before writing anything.
    plans, split_hashes, stats = [], {}, {}
    for split in ("train", "val"):
        split_hashes[split] = set()
        stats[split] = {"source_images": 0, "source_points": 0,
                        "flower_crops": 0, "background_samples": 0}
        for image_path in sorted((source / "images" / split).glob("*")):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            # Pixel hashes also detect identical images with different metadata.
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            split_hashes[split].add(digest)
            width, height = image.size
            boxes = read_crop_boxes(source / "labels" / split / (image_path.stem + ".txt"),
                                    width, height)
            entry = entries.get(image_path.name)
            if boxes and entry is None:
                raise ValueError(f"Flower image has no tip annotation entry: {image_path.name}")
            if entry and entry["size"] != image.size:
                raise ValueError(f"XML/image dimensions disagree: {image_path.name}")
            points = entry["points"] if entry else []
            if points and not boxes:
                raise ValueError(f"Tips present but no lily box: {image_path.name}")
            covered = {i for box in boxes for i, (x, y) in enumerate(points)
                       if box[0] <= x < box[2] and box[1] <= y < box[3]}
            if len(covered) != len(points):
                raise ValueError(f"Some tips fall outside every crop: {image_path.name}")
            stats[split]["source_images"] += 1
            stats[split]["source_points"] += len(points)
            stats[split]["flower_crops"] += len(boxes)
            stats[split]["background_samples"] += int(not boxes)
            # Each background photograph becomes one zero-target training example.
            for index, box in enumerate(boxes or [[0, 0, width, height]]):
                plans.append((split, image_path, box, index, points,
                              "flower" if boxes else "background", digest))
    if split_hashes["train"] & split_hashes["val"]:
        raise ValueError("Identical source image pixels occur in train and val.")
    if any(stats[s]["source_images"] == 0 for s in stats):
        raise ValueError("Expected nonempty train and val folders.")

    records, cards = [], []
    for split, image_path, box, index, points, kind, digest in plans:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        canvas, transformed, indices, geometry = prepare_sample(image, box, points)
        target = make_heatmap(transformed)
        name = f"{image_path.stem}__{kind}_{index:02d}"
        image_relative = Path(split) / "images" / f"{name}.png"
        target_relative = Path(split) / "heatmaps" / f"{name}.npy"
        review_relative = Path("review") / f"{split}__{name}.jpg"
        for relative in (image_relative, target_relative, review_relative):
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output / image_relative)
        np.save(output / target_relative, target)
        records.append({"id": name, "split": split, "kind": kind,
                        "source_image": image_path.name, "source_sha256": digest,
                        "source_wh": list(image.size), "image": str(image_relative),
                        "heatmap": str(target_relative), "points_xy": transformed,
                        "source_point_indices": indices,
                        "source_points_xy": [points[i] for i in indices], **geometry})

        # Review marks are ONLY on a separate image, never the training input.
        review = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE))
        review.paste(canvas, (0, 0))
        aligned = canvas.copy()
        draw = ImageDraw.Draw(aligned)
        for x, y in transformed:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow", outline="black")
        review.paste(aligned, (IMAGE_SIZE, 0))
        review.save(output / review_relative)
        cards.append(f'<figure><img loading="lazy" src="{review_relative}">'
                     f'<figcaption>{html.escape(name)} · {split} · '
                     f'{len(transformed)} tips</figcaption></figure>')

    for split in stats:
        selected = [r for r in records if r["split"] == split]
        stats[split]["samples"] = len(selected)
        stats[split]["zero_target_samples"] = sum(not r["points_xy"] for r in selected)
        stats[split]["crop_point_instances"] = sum(len(r["points_xy"]) for r in selected)
    manifest = {"image_size": IMAGE_SIZE, "sigma": SIGMA, "crop_margin": CROP_MARGIN,
                "target_channels": ["anther_tip"], "source_dataset": str(source),
                "annotation_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in xml_files},
                "statistics": stats, "samples": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Anther crop review</title>'
        '<style>body{font:16px system-ui;background:#eee}figure{display:inline-block;'
        'margin:12px;background:white;padding:10px}img{max-width:100%}'
        'figcaption{max-width:512px;overflow-wrap:anywhere}</style>'
        '<h1>Anther crop review</h1><p>Left: training input. Right: annotated tip '
        'centers. Zero-tip crops have all-zero heatmaps. Overlapping flower crops '
        'can contain the same source tip; they stay in the same split.</p>' + "".join(cards))
    print(json.dumps(stats, indent=2))
    print("Dataset saved to:", output)
    print("Open in your browser:", output / "review.html")


if __name__ == "__main__":
    main()
