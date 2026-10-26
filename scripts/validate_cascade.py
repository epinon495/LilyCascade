"""Save in LilyCascade/scripts beside heatmap_tools.py; run in PyCharm.

Evaluate complete-image anther predictions using unique CVAT point annotations.
Detect flowers once; infer each heatmap checkpoint once per crop; reuse maps for
all thresholds. Training images and the test set are never evaluated here.
"""

from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import ultralytics

from heatmap_tools import TinyUNet, find_peaks


# ------------------------- Experiment settings -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LILY_RUN = "lily_n416_baseline"
HEATMAP_RUNS = {
    "baseline": "anther_unet256_baseline",
    "augmented": "anther_unet256_augmented_additionalepochs",
}
LILY_IMAGE_SIZE = 416
LILY_CONFIDENCE = 0.60
CROP_MARGIN = 0.10
HEATMAP_SIZE = 256
PEAK_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MIN_PEAK_DISTANCE = 4
MATCH_RADIUS = 6.0
REFERENCE_MARGIN = 0.10
DEVICE = "cpu"
CPU_THREADS = 2
# MATCH_RADIUS is measured in a fixed, annotation-defined 256-pixel flower
# crop, NOT in a predicted crop or in meters. See reference_points().
# -----------------------------------------------------------------------


def prepare_crop(image, box, margin=CROP_MARGIN):
    """Same preprocessing as the cascade: expanded box, resize, gray padding."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    left = max(0, math.floor(x0 - margin * width))
    top = max(0, math.floor(y0 - margin * height))
    right = min(image.width, math.ceil(x1 + margin * width))
    bottom = min(image.height, math.ceil(y1 + margin * height))
    if right <= left or bottom <= top:
        raise ValueError("Empty flower crop.")
    cropped = image.crop((left, top, right, bottom))
    scale = HEATMAP_SIZE / max(cropped.size)
    rw, rh = max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))
    px, py = (HEATMAP_SIZE - rw) // 2, (HEATMAP_SIZE - rh) // 2
    canvas = Image.new("RGB", (HEATMAP_SIZE, HEATMAP_SIZE), (114, 114, 114))
    canvas.paste(cropped.resize((rw, rh), Image.Resampling.BILINEAR), (px, py))
    return canvas, {"crop_xyxy": [left, top, right, bottom],
                    "resized_wh": [rw, rh], "padding_xy": [px, py],
                    "scale_xy": [rw / cropped.width, rh / cropped.height]}


def read_annotations(folder):
    entries = {}
    files = sorted(folder.glob("*.xml"))
    if not files:
        raise ValueError(f"No XML annotations found in {folder}")
    for file in files:
        for item in ET.parse(file).getroot().findall("image"):
            name = item.get("name")
            if name in entries:
                raise ValueError(f"Duplicate XML entry: {name}")
            width, height = int(item.get("width")), int(item.get("height"))
            points = []
            for shape in item.findall("points"):
                if shape.get("label") != "anther_tip":
                    raise ValueError(f"Unexpected point label in {name}")
                for pair in shape.get("points").split(";"):
                    x, y = map(float, pair.split(","))
                    if not (math.isfinite(x) and math.isfinite(y)
                            and 0 <= x < width and 0 <= y < height):
                        raise ValueError(f"Invalid point in {name}")
                    points.append([x, y])
            entries[name] = {"size": (width, height), "points": points}
    return entries, files


def read_lily_boxes(path, width, height):
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        if cls != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Expected valid lily-only boxes in {path}")
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def reference_points(image, boxes, points):
    """Give each unique point a fixed localization tolerance.

    Choose the smallest padded annotated flower box containing the point.
    Its resize scales define a six-pixel tolerance ellipse in the source frame.
    This reference stays independent of YOLO's predictions and peak threshold:
    predicting an oversized flower box cannot make matching more forgiving.
    It is an evaluation reference, not a claim of point-to-flower ownership.
    """
    geometries = [prepare_crop(image, box, REFERENCE_MARGIN)[1] for box in boxes]
    references = []
    for index, (x, y) in enumerate(points):
        containing = [g for g in geometries if
                      g["crop_xyxy"][0] <= x < g["crop_xyxy"][2]
                      and g["crop_xyxy"][1] <= y < g["crop_xyxy"][3]]
        if not containing:
            raise ValueError(f"Annotated tip {index} falls outside all reference flower crops.")
        chosen = min(containing, key=lambda g:
                     (g["crop_xyxy"][2] - g["crop_xyxy"][0]) *
                     (g["crop_xyxy"][3] - g["crop_xyxy"][1]))
        references.append({"id": index, "xy": [x, y], "scale_xy": chosen["scale_xy"]})
    return references


def extract_candidates(heatmap, geometry, threshold, flower_number):
    px, py = geometry["padding_xy"]
    rw, rh = geometry["resized_wh"]
    masked = np.zeros_like(heatmap)
    masked[py:py + rh, px:px + rw] = heatmap[py:py + rh, px:px + rw]
    left, top, _, _ = geometry["crop_xyxy"]
    sx, sy = geometry["scale_xy"]
    return [{"xy": [(x - px) / sx + left, (y - py) / sy + top],
             "score": score, "flower_number": flower_number}
            for x, y, score in find_peaks(masked, threshold, MIN_PEAK_DISTANCE)]


def match_candidates(candidates, references):
    """One-to-one, confidence-ordered greedy matching in each photograph.

    No cross-crop merging: repeated detections of a tip cannot earn another TP;
    they remain FPs. This avoids merging nearby but genuinely distinct tips.
    """
    remaining = set(range(len(references)))
    pairs, extra = [], []
    for prediction_index in sorted(range(len(candidates)),
                                    key=lambda i: -candidates[i]["score"]):
        prediction = candidates[prediction_index]

        def distance(index):
            target = references[index]
            return math.hypot((prediction["xy"][0] - target["xy"][0]) * target["scale_xy"][0],
                              (prediction["xy"][1] - target["xy"][1]) * target["scale_xy"][1])

        closest = min(remaining, key=distance) if remaining else None
        if closest is not None and distance(closest) <= MATCH_RADIUS:
            pairs.append({"prediction": prediction_index, "target": closest,
                          "error_reference_px": distance(closest)})
            remaining.remove(closest)
        else:
            extra.append(prediction_index)
    return {"tp": len(pairs), "fp": len(extra), "fn": len(remaining),
            "pairs": pairs, "extra_indices": extra, "missed_indices": sorted(remaining)}


def scores(tp, fp, fn):
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def save_overlay(frame, evaluation, path):
    with Image.open(frame["source"]) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    radius, line = max(5, round(max(image.size) / 350)), max(2, round(max(image.size) / 900))
    for box in frame["lily_boxes"]:
        draw.rectangle(box, outline="#169eff", width=line)
    missed = set(evaluation["missed_indices"])
    for index, target in enumerate(frame["references"]):
        x, y = target["xy"]
        color = "red" if index in missed else "#25dc63"
        draw.line((x - radius, y, x + radius, y), fill=color, width=line)
        draw.line((x, y - radius, x, y + radius), fill=color, width=line)
    extra = set(evaluation["extra_indices"])
    for index, prediction in enumerate(evaluation["predictions"]):
        x, y = prediction["xy"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline="orange" if index in extra else "cyan", width=line)
    image.thumbnail((2000, 2000))
    image.save(path, quality=93)


def main(project_root=None, output_root=None):
    if DEVICE == "cpu":
        torch.set_num_threads(CPU_THREADS)
    root = Path(project_root or PROJECT_ROOT)
    dataset = root / "datasets/lily_v1"
    entries, annotation_files = read_annotations(root / "datasets/annotations/anther_endpoints")
    files = sorted(p for p in (dataset / "images/val").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files or not PEAK_THRESHOLDS or not all(0 < t <= 1 for t in PEAK_THRESHOLDS):
        raise ValueError("Need validation photographs and peak thresholds in (0, 1].")
    if MATCH_RADIUS <= 0 or MIN_PEAK_DISTANCE < 0:
        raise ValueError("Invalid point matching or suppression distance.")
    # Preflight missing annotations before doing expensive inference.
    frames = []
    for file in files:
        with Image.open(file) as opened:
            image = opened.convert("RGB")
        boxes = read_lily_boxes(dataset / "labels/val" / (file.stem + ".txt"), *image.size)
        entry = entries.get(file.name)
        if boxes and entry is None:
            raise ValueError(f"No endpoint annotation entry for {file.name}")
        if entry and entry["size"] != image.size:
            raise ValueError(f"Image and XML dimensions disagree: {file.name}")
        points = entry["points"] if entry else []
        frames.append({"image": file.name, "source": str(file), "size_wh": list(image.size),
                       "references": reference_points(image, boxes, points)})
    lily_path = root / "runs" / LILY_RUN / "weights/best.pt"
    lily_model = YOLO(lily_path)
    models, checkpoint_info = {}, {}
    for name, run in HEATMAP_RUNS.items():
        path = root / "runs" / run / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = TinyUNet(width=checkpoint["config"]["MODEL_WIDTH"])
        network.load_state_dict(checkpoint["model_state"])
        models[name] = network.to(DEVICE).eval()
        checkpoint_info[name] = {"path": str(path), "epoch": checkpoint["epoch"],
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(output_root or root / "runs" / f"cascade_validation_{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    cache_folder = output / "heatmap_cache"
    cache_folder.mkdir()
    # Keep all heatmaps in memory for this small dataset and also save them.
    cache = {}
    for frame in frames:
        result = lily_model.predict(source=frame["source"], imgsz=LILY_IMAGE_SIZE,
                                    conf=LILY_CONFIDENCE, device=DEVICE, verbose=False)[0]
        image = Image.fromarray(result.orig_img[:, :, ::-1].copy())
        if list(image.size) != frame["size_wh"]:
            raise ValueError(f"YOLO/XML orientation mismatch: {frame['image']}")
        frame["lily_boxes"] = result.boxes.xyxy.cpu().tolist()
        frame["lily_scores"] = result.boxes.conf.cpu().tolist()
        frame["crops"] = []
        for number, box in enumerate(frame["lily_boxes"]):
            prepared, geometry = prepare_crop(image, box)
            frame["crops"].append(geometry)
            array = np.asarray(prepared, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(DEVICE)
            for name, network in models.items():
                with torch.inference_mode():
                    heatmap = network(tensor).sigmoid()[0, 0].cpu().numpy()
                cache[(frame["image"], number, name)] = heatmap
                np.save(cache_folder / f"{Path(frame['image']).stem}_{number:02d}_{name}.npy", heatmap)
        # Geometric coverage is a diagnostic, not proof a tip is recognizable.
        # Matching tolerance can accept a prediction near a crop's boundary.
        frame["covered_targets"] = sum(any(
            g["crop_xyxy"][0] <= p["xy"][0] < g["crop_xyxy"][2]
            and g["crop_xyxy"][1] <= p["xy"][1] < g["crop_xyxy"][3]
            for g in frame["crops"]) for p in frame["references"])
        print(f"Prepared {frame['image']}: {len(frame['crops'])} detected crops", flush=True)

    rows, evaluations = [], {}
    for name in models:
        for threshold in PEAK_THRESHOLDS:
            individual = []
            for frame in frames:
                predictions = []
                for number, geometry in enumerate(frame["crops"]):
                    predictions.extend(extract_candidates(cache[(frame["image"], number, name)],
                                                         geometry, threshold, number))
                counts = match_candidates(predictions, frame["references"])
                individual.append({"image": frame["image"], "predictions": predictions, **counts})
            tp, fp, fn = (sum(item[key] for item in individual) for key in ("tp", "fp", "fn"))
            errors = [pair["error_reference_px"] for item in individual for pair in item["pairs"]]
            rows.append({"model": name, "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                         **scores(tp, fp, fn), "matched_error_reference_px":
                         float(np.mean(errors)) if errors else None})
            evaluations[(name, threshold)] = individual
    best = {name: max((row for row in rows if row["model"] == name),
                      key=lambda row: (row["f1"], row["precision"])) for name in models}
    total_targets = sum(len(frame["references"]) for frame in frames)
    covered = sum(frame["covered_targets"] for frame in frames)
    settings = {"lily_checkpoint": str(lily_path), "lily_image_size": LILY_IMAGE_SIZE,
                "lily_confidence": LILY_CONFIDENCE, "crop_margin": CROP_MARGIN,
                "heatmap_size": HEATMAP_SIZE, "minimum_peak_distance": MIN_PEAK_DISTANCE,
                "matching_radius_reference_px": MATCH_RADIUS, "reference_margin": REFERENCE_MARGIN,
                "thresholds": PEAK_THRESHOLDS, "device": DEVICE, "cpu_threads": CPU_THREADS,
                "torch_version": str(torch.__version__), "ultralytics_version": ultralytics.__version__}
    notes = ["Validation only; these settings were selected using this validation set.",
             "All source annotations count once per photograph, even outside accepted YOLO crops.",
             "Scores use unique source tips, not repeated crop instances; not directly comparable to training F1.",
             "Repeated predictions of an already matched tip count as false positives; no cross-crop merging.",
             "Matching is confidence-ordered greedy, within six reference-crop pixels, not a physical tolerance.",
             "Reference scale uses the smallest padded annotated flower box containing each tip.",
             "Do not interpret matched-point error without precision and recall.",
             "Every annotated tip is scored; occluded/background-view cases are not excluded."]
    report = {"settings": settings, "checkpoints": checkpoint_info,
              "annotation_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in annotation_files},
              "source_images": len(frames), "unique_annotated_tips": total_targets,
              "tips_inside_detector_crops": covered, "tips_outside_detector_crops": total_targets - covered,
              "rows": rows, "best_by_validation_f1": best, "notes": notes}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    (output / "frames.json").write_text(json.dumps(frames, indent=2))
    serializable = {f"{name}_{threshold:.2f}": value for (name, threshold), value in evaluations.items()}
    (output / "point_evaluations.json").write_text(json.dumps(serializable))
    lines = ["# Complete cascade validation", "",
             f"{len(frames)} photographs; {total_targets} unique annotated tips; "
             f"{covered} tips inside accepted detector crops.", "",
             "| Model | Peak threshold | Correct | Extra | Missed | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['threshold']:.2f} | {row['tp']} | {row['fp']} | "
                     f"{row['fn']} | {row['precision']:.1%} | {row['recall']:.1%} | {row['f1']:.1%} |")
    lines += ["", *[f"- {note}" for note in notes]]
    (output / "comparison.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for name in models:
        selected = [row for row in rows if row["model"] == name]
        for ax, metric in zip(axes, ("precision", "recall", "f1")):
            ax.plot(PEAK_THRESHOLDS, [row[metric] for row in selected], marker="o", label=name)
            ax.set_xlabel("Heatmap peak threshold")
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "threshold_curves.png", dpi=160)
    plt.close(fig)
    cards = []
    overlay_folder = output / "overlays"
    overlay_folder.mkdir()
    for name, chosen in best.items():
        cards.append(f'<h2>{html.escape(name)}: best validation F1 at threshold {chosen["threshold"]:.2f}</h2>')
        for frame, evaluation in zip(frames, evaluations[(name, chosen["threshold"])]):
            relative = f"overlays/{name}_{Path(frame['image']).stem}.jpg"
            save_overlay(frame, evaluation, output / relative)
            cards.append(f'<article><h3>{html.escape(frame["image"])}</h3>'
                         f'<p>{evaluation["tp"]} correct; {evaluation["fp"]} extra; '
                         f'{evaluation["fn"]} missed.</p><a href="{relative}">'
                         f'<img loading="lazy" src="{relative}"></a></article>')
    page = '''<!doctype html><meta charset="utf-8"><title>Cascade validation</title>
<style>body{font:16px system-ui;background:#eee;max-width:1300px;margin:25px auto;padding:15px}
article{background:white;padding:15px;margin:15px 0}img{max-width:100%}table{border-collapse:collapse;background:white}
td,th{border:1px solid #aaa;padding:8px}</style><h1>Complete cascade validation</h1>'''
    page += f'<p>{len(frames)} photographs · {total_targets} unique annotated tips · {covered} inside accepted YOLO crops.</p>'
    page += '<p>Green cross: matched annotation. Red cross: missed annotation. Cyan circle: matched prediction. '
    page += 'Orange circle: extra prediction. Blue box: YOLO detection. Preview images are reduced for display.</p>'
    page += '<table><tr><th>Model</th><th>Threshold</th><th>Correct</th><th>Extra</th><th>Missed</th><th>P</th><th>R</th><th>F1</th></tr>'
    for row in rows:
        page += (f'<tr><td>{html.escape(row["model"])}</td><td>{row["threshold"]:.2f}</td>'
                 f'<td>{row["tp"]}</td><td>{row["fp"]}</td><td>{row["fn"]}</td>'
                 f'<td>{row["precision"]:.1%}</td><td>{row["recall"]:.1%}</td><td>{row["f1"]:.1%}</td></tr>')
    page += '</table><p>These are validation-selected settings, not an independent final test.</p>'
    page += '<p>Duplicate predictions count as extras. Matching tolerance is defined by annotations; no targets are dropped because YOLO missed them.</p>'
    page += '<img src="threshold_curves.png">' + "".join(cards)
    (output / "review.html").write_text(page)
    # Retain the evaluator and model definition used to produce the report.
    for file in (Path(__file__), Path(__file__).with_name("heatmap_tools.py")):
        if file.is_file():
            (output / file.name).write_bytes(file.read_bytes())
    print("\n".join(lines))
    print("Open:", output / "review.html")


if __name__ == "__main__":
    main()
