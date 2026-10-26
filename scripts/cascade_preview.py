from pathlib import Path
import math
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from heatmap_tools import TinyUNet

project_dir = Path(__file__).resolve().parents[1]

heatmap_model_path = (
    project_dir
    / "runs"
    / "anther_unet256_baseline"
    / "best.pt"
)

lily_model_path = (project_dir / "runs" / "lily_n416_baseline" / "weights"  / "best.pt")

image_path = (project_dir / "datasets" / "lily_v1" / "images" / "val" / "session02__IMG_8038.jpg")

output_dir = project_dir / "runs" / "cascade_baseline_conf60"

lily_imgsz = 416
lily_conf = 0.6
CROP_MARGIN = 0.1
HEATMAP_IMAGE_SIZE = 256

def prepare_flower_crop(image, box):
    """Return the prepared image and geometry needed to map points back."""
    x0, y0, x1, y1 = box

    box_width = x1 - x0
    box_height = y1 - y0

    # Expand the detected box, staying inside the photograph.
    left = max(0, math.floor(x0 - CROP_MARGIN * box_width))
    top = max(0, math.floor(y0 - CROP_MARGIN * box_height))

    right = min(
        image.width,
        math.ceil(x1 + CROP_MARGIN * box_width),
    )
    bottom = min(
        image.height,
        math.ceil(y1 + CROP_MARGIN * box_height),
    )

    if right <= left or bottom <= top:
        raise ValueError("The detected box produces an empty crop.")

    crop = image.crop((left, top, right, bottom))

    # Resize proportionally: the longest side becomes 256 pixels.
    scale = HEATMAP_IMAGE_SIZE / max(crop.size)

    resized_width = max(1, round(crop.width * scale))
    resized_height = max(1, round(crop.height * scale))

    resized = crop.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    # Center the resized image on a square gray canvas.
    pad_x = (HEATMAP_IMAGE_SIZE - resized_width) // 2
    pad_y = (HEATMAP_IMAGE_SIZE - resized_height) // 2

    prepared = Image.new(
        "RGB",
        (HEATMAP_IMAGE_SIZE, HEATMAP_IMAGE_SIZE),
        (114, 114, 114),
    )
    prepared.paste(resized, (pad_x, pad_y))

    geometry = {
        "crop_xyxy": (left, top, right, bottom),
        "scale_xy": (
            resized_width / crop.width,
            resized_height / crop.height,
        ),
        "padding_xy": (pad_x, pad_y),
    }

    return prepared, geometry

def main():
    if not lily_model_path.is_file():
        raise FileNotFoundError(lily_model_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    lily_model = YOLO(lily_model_path)

    if not heatmap_model_path.is_file():
        raise FileNotFoundError(heatmap_model_path)

    checkpoint = torch.load(
        heatmap_model_path,
        map_location="cpu",
        weights_only=True,
    )

    heatmap_model = TinyUNet(
        width=checkpoint["config"]["MODEL_WIDTH"]
    )

    heatmap_model.load_state_dict(
        checkpoint["model_state"]
    )

    heatmap_model.eval()

    results = lily_model.predict(
        source=str(image_path),
        imgsz=lily_imgsz,
        conf=lily_conf,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    print("Original image dimensions:", result.orig_shape)
    print("Number of detected lilies:", len(result.boxes))

    with Image.open(image_path) as opened:
        original_image = opened.convert("RGB")

    for flower_number, box in enumerate(result.boxes):
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()

        prepared_crop, geometry = prepare_flower_crop(
            original_image,
            (x0, y0, x1, y1),
        )

        crop_path = (
            output_dir / f"flower_{flower_number:02d}_input.png"
        )

        prepared_crop.save(crop_path)

        # Match the image normalization used during training.
        image_array = (
            np.asarray(prepared_crop, dtype=np.float32) / 255.0
        )

        # Change height × width × channels to channels × height × width.
        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # Add a batch dimension: one image.
        image_tensor = image_tensor.unsqueeze(0)

        with torch.inference_mode():
            logits = heatmap_model(image_tensor)
            predicted_heatmap = logits.sigmoid()[0, 0].numpy()

        np.save(
            output_dir / f"flower_{flower_number:02d}_heatmap.npy",
            predicted_heatmap,
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(prepared_crop)
        axes[0].set_title("Flower crop")

        axes[1].imshow(
            predicted_heatmap,
            cmap="hot",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Predicted anther heatmap")

        axes[2].imshow(prepared_crop)
        axes[2].imshow(
            np.ma.masked_less(predicted_heatmap, 0.05),
            cmap="hot",
            vmin=0,
            vmax=1,
            alpha=0.7,
        )
        axes[2].set_title("Prediction overlay")

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        fig.savefig(
            output_dir / f"flower_{flower_number:02d}_prediction.png",
            dpi=150,
        )

        plt.close(fig)

        print("Saved crop:", crop_path)
        print("Crop geometry:", geometry)

        print(
            f"Flower {flower_number}: "
            f"confidence={confidence:.3f}, "
            f"box=({x0:.1f}, {y0:.1f}, "
            f"{x1:.1f}, {y1:.1f})"
        )

    result.save(
        filename=str(output_dir / "lily_detections.jpg")
    )


if __name__ == "__main__":
    main()