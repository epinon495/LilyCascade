"""Save in LilyCascade/scripts beside heatmap_tools.py, then run in PyCharm.

Detect lilies, predict anther heatmaps, extract peaks, and reverse the crop
geometry to report pixel coordinates in the original photograph. No training.
"""

from datetime import datetime
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from heatmap_tools import TinyUNet, find_peaks


# -------------------- Change experiment settings here --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = Path("datasets/lily_v1/images/val/session02__IMG_8038.jpg")
LILY_RUN = "lily_n416_baseline"
HEATMAP_RUN = "anther_unet256_baseline"
# To compare the original model, set HEATMAP_RUN = "anther_unet256_baseline".
LILY_IMAGE_SIZE = 416
LILY_CONFIDENCE = 0.60
CROP_MARGIN = 0.10
HEATMAP_IMAGE_SIZE = 256
PEAK_THRESHOLD = 0.30
MIN_PEAK_DISTANCE = 4  # Pixels in the 256 x 256 heatmap, not original pixels.
DEVICE = "cpu"  # Enough for single-photo inspection; no GPU training conflict.
# Every invocation gets a fresh output folder, preserving earlier comparisons.
# -------------------------------------------------------------------------


def prepare_flower_crop(image, box):
    """Use the same margin, resize, and square padding as heatmap training."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    left = max(0, math.floor(x0 - CROP_MARGIN * width))
    top = max(0, math.floor(y0 - CROP_MARGIN * height))
    right = min(image.width, math.ceil(x1 + CROP_MARGIN * width))
    bottom = min(image.height, math.ceil(y1 + CROP_MARGIN * height))
    if right <= left or bottom <= top:
        raise ValueError("Detected box produces an empty crop.")
    crop = image.crop((left, top, right, bottom))
    scale = HEATMAP_IMAGE_SIZE / max(crop.size)
    resized_width = max(1, round(crop.width * scale))
    resized_height = max(1, round(crop.height * scale))
    pad_x = (HEATMAP_IMAGE_SIZE - resized_width) // 2
    pad_y = (HEATMAP_IMAGE_SIZE - resized_height) // 2
    canvas = Image.new("RGB", (HEATMAP_IMAGE_SIZE, HEATMAP_IMAGE_SIZE), (114, 114, 114))
    canvas.paste(crop.resize((resized_width, resized_height),
                             Image.Resampling.BILINEAR), (pad_x, pad_y))
    geometry = {"crop_xyxy": [left, top, right, bottom],
                "scale_xy": [resized_width / crop.width, resized_height / crop.height],
                "padding_xy": [pad_x, pad_y],
                "resized_wh": [resized_width, resized_height]}
    return canvas, geometry


def extract_points(heatmap, geometry):
    """Ignore gray padding, then find peaks and map them to original pixels."""
    pad_x, pad_y = geometry["padding_xy"]
    resized_width, resized_height = geometry["resized_wh"]
    valid = np.zeros(heatmap.shape, dtype=bool)
    valid[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = True
    masked = np.where(valid, heatmap, 0).astype(np.float32)
    peaks = find_peaks(masked, PEAK_THRESHOLD, MIN_PEAK_DISTANCE)
    left, top, _, _ = geometry["crop_xyxy"]
    scale_x, scale_y = geometry["scale_xy"]
    points = []
    for index, (x, y, score) in enumerate(peaks, start=1):
        # Reverse padding, resize, and finally crop translation.
        original_x = (x - pad_x) / scale_x + left
        original_y = (y - pad_y) / scale_y + top
        points.append({"point_number": index, "heatmap_xy": [x, y],
                       "original_xy": [original_x, original_y],
                       "heatmap_score": score})
    return points


def save_crop_review(crop, heatmap, points, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(crop)
    axes[0].set_title("YOLO flower crop")
    axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Predicted heatmap")
    axes[2].imshow(crop)
    axes[2].set_title(f"Accepted peaks: {len(points)} (score >= {PEAK_THRESHOLD})")
    for point in points:
        x, y = point["heatmap_xy"]
        axes[2].scatter(x, y, s=70, facecolors="none", edgecolors="cyan")
        axes[2].annotate(str(point["point_number"]), (x, y), xytext=(4, -5),
                         textcoords="offset points", color="yellow", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_font(size):
    path = Path("/System/Library/Fonts/Helvetica.ttc")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def main(project_root=None, output_root=None):
    root = Path(project_root or PROJECT_ROOT)
    image_path = root / IMAGE_PATH  # Absolute IMAGE_PATH also works.
    lily_path = root / "runs" / LILY_RUN / "weights/best.pt"
    heatmap_path = root / "runs" / HEATMAP_RUN / "best.pt"
    for path in (image_path, lily_path, heatmap_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < PEAK_THRESHOLD <= 1 or MIN_PEAK_DISTANCE < 0:
        raise ValueError("Use a peak threshold in (0, 1] and nonnegative minimum distance.")
    checkpoint = torch.load(heatmap_path, map_location="cpu", weights_only=True)
    model = TinyUNet(width=checkpoint["config"]["MODEL_WIDTH"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(output_root or root / "runs" / f"cascade_points_{HEATMAP_RUN}_{stamp}")
    output.mkdir(parents=True, exist_ok=False)

    lily_model = YOLO(lily_path)
    result = lily_model.predict(source=str(image_path), imgsz=LILY_IMAGE_SIZE,
                                conf=LILY_CONFIDENCE, device=DEVICE, verbose=False)[0]
    # Use YOLO's decoded original image, ensuring box/image orientation agrees.
    original = Image.fromarray(result.orig_img[:, :, ::-1].copy())
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(max(original.size) / 650))
    font = get_font(max(12, round(max(original.size) / 160)))
    radius = max(5, round(max(original.size) / 400))
    flowers = []
    for flower_number, box in enumerate(result.boxes):
        box_xyxy = box.xyxy[0].cpu().tolist()
        lily_score = float(box.conf[0])
        crop, geometry = prepare_flower_crop(original, box_xyxy)
        array = np.asarray(crop, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(DEVICE)
        with torch.inference_mode():
            heatmap = model(tensor).sigmoid()[0, 0].cpu().numpy()
        points = extract_points(heatmap, geometry)
        prefix = f"flower_{flower_number:02d}"
        crop.save(output / f"{prefix}_input.png")
        np.save(output / f"{prefix}_heatmap.npy", heatmap)
        save_crop_review(crop, heatmap, points, output / f"{prefix}_points.png")
        draw.rectangle(box_xyxy, outline="#169eff", width=line_width)
        draw.text((box_xyxy[0], max(0, box_xyxy[1] - font.size - 3)),
                  f"F{flower_number} lily {lily_score:.2f}", fill="#169eff",
                  font=font, stroke_width=1, stroke_fill="black")
        for point in points:
            point["id"] = f"F{flower_number}:P{point['point_number']}"
            x, y = point["original_xy"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline="cyan", width=line_width)
            draw.text((x + radius, y), point["id"], fill="yellow", font=font,
                      stroke_width=1, stroke_fill="black")
        flowers.append({"flower_number": flower_number, "lily_score": lily_score,
                        "detected_box_xyxy": box_xyxy, "geometry": geometry,
                        "points": points})
        print(f"Flower {flower_number}: {len(points)} candidate tips")

    overlay.save(output / "full_image_points.jpg", quality=95)
    report = {"source_image": str(image_path), "original_wh": list(original.size),
              "lily_checkpoint": str(lily_path), "heatmap_checkpoint": str(heatmap_path),
              "heatmap_checkpoint_epoch": checkpoint["epoch"],
              "settings": {"lily_image_size": LILY_IMAGE_SIZE,
                           "lily_confidence": LILY_CONFIDENCE, "crop_margin": CROP_MARGIN,
                           "heatmap_image_size": HEATMAP_IMAGE_SIZE,
                           "peak_threshold": PEAK_THRESHOLD,
                           "minimum_peak_distance": MIN_PEAK_DISTANCE, "device": DEVICE},
              "notes": ["Heatmap scores are not calibrated probabilities.",
                        "Coordinates are original-image pixels, not physical positions.",
                        "Points are candidates in each crop, not guaranteed anther identities.",
                        "Overlapping crops may repeat a physical tip; no cross-crop merging is applied."],
              "flowers": flowers}
    (output / "tip_coordinates.json").write_text(json.dumps(report, indent=2))
    if not flowers:
        print("No lilies passed the confidence threshold; saved an empty point report.")
    print("Heatmap checkpoint epoch:", checkpoint["epoch"])
    print("Open:", output / "full_image_points.jpg")
    print("Coordinates:", output / "tip_coordinates.json")


if __name__ == "__main__":
    main()
