from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_NAME = "session01__IMG_5573.jpg"
IMAGE_SIZE = 256
CROP_MARGIN = 0.10
SIGMA = 2.0

DATASET = PROJECT_ROOT / "datasets" / "lily_v1"

ANNOTATIONS = (
    PROJECT_ROOT
    / "datasets"
    / "annotations"
    / "anther_endpoints"
    / "batch_01.xml"
)

OUTPUT = PROJECT_ROOT / "runs" / "heatmap_preview"


def make_heatmap(points, size, sigma):
    """Create Gaussian peaks centered on the supplied coordinates."""
    yy, xx = np.mgrid[0:size, 0:size]

    heatmap = np.zeros((size, size), dtype=np.float32)

    for x, y in points:
        squared_distance = (xx - x) ** 2 + (yy - y) ** 2

        gaussian = np.exp(
            -squared_distance / (2 * sigma ** 2)
        ).astype(np.float32)

        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


def main():
    image_path = DATASET / "images" / "train" / IMAGE_NAME

    label_path = (
        DATASET / "labels" / "train" / f"{image_path.stem}.txt"
    )

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Find this photograph's endpoint annotations.
    root = ET.parse(ANNOTATIONS).getroot()

    entry = next(
        item for item in root.findall("image")
        if item.get("name") == IMAGE_NAME
    )

    points = []

    for annotation in entry.findall("points"):
        if annotation.get("label") == "anther_tip":
            for coordinate in annotation.get("points").split(";"):
                x, y = map(float, coordinate.split(","))
                points.append((x, y))

    # This example has one lily box.
    label_lines = label_path.read_text().splitlines()

    if len(label_lines) != 1:
        raise ValueError("Choose an image with exactly one lily box.")

    class_id, cx, cy, bw, bh = map(
        float, label_lines[0].split()
    )

    # Convert normalized YOLO coordinates into source-image pixels.
    box_width = bw * width
    box_height = bh * height

    left = (cx * width) - box_width / 2
    top = (cy * height) - box_height / 2

    # Expand the box and keep it inside the photograph.
    x0 = max(0, int(np.floor(left - CROP_MARGIN * box_width)))
    y0 = max(0, int(np.floor(top - CROP_MARGIN * box_height)))

    x1 = min(
        width,
        int(np.ceil(left + (1 + CROP_MARGIN) * box_width)),
    )
    y1 = min(
        height,
        int(np.ceil(top + (1 + CROP_MARGIN) * box_height)),
    )

    crop = image.crop((x0, y0, x1, y1))

    # Resize proportionally, then pad to a square.
    scale = IMAGE_SIZE / max(crop.size)

    resized_width = round(crop.width * scale)
    resized_height = round(crop.height * scale)

    resized = crop.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    pad_x = (IMAGE_SIZE - resized_width) // 2
    pad_y = (IMAGE_SIZE - resized_height) // 2

    model_input = Image.new(
        "RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114)
    )
    model_input.paste(resized, (pad_x, pad_y))

    # Apply the same crop, resize, and padding to the points.
    transformed_points = []

    for x, y in points:
        if not (x0 <= x < x1 and y0 <= y < y1):
            raise ValueError("An annotated tip falls outside this crop.")

        new_x = (x - x0) * resized_width / crop.width + pad_x
        new_y = (y - y0) * resized_height / crop.height + pad_y

        transformed_points.append((new_x, new_y))

    heatmap = make_heatmap(
        transformed_points, IMAGE_SIZE, SIGMA
    )

    # Save both the input and the numerical training target.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_input.save(OUTPUT / "input.png")
    np.save(OUTPUT / "target_heatmap.npy", heatmap)

    # Display the input, target, and their alignment.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(model_input)
    axes[0].set_title("Flower crop")

    axes[1].imshow(
        heatmap, cmap="hot", vmin=0, vmax=1
    )
    axes[1].set_title("Target anther heatmap")

    axes[2].imshow(model_input)
    axes[2].imshow(
        np.ma.masked_less(heatmap, 0.05),
        cmap="hot",
        vmin=0,
        vmax=1,
        alpha=0.7,
    )
    axes[2].set_title("Target alignment")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT / "preview.png", dpi=180)
    plt.close(fig)

    print("Annotated endpoints:", len(transformed_points))
    print("Input size:", model_input.size)
    print("Heatmap shape:", heatmap.shape)
    print("Preview saved to:", OUTPUT / "preview.png")


if __name__ == "__main__":
    main()