from pathlib import Path
import shutil

import yaml


# Find the project folder from this script's location.
# This script lives inside LilyCascade/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Read the original three-class dataset from the archive.
SOURCE = Path(
    "/Users/evpinon/FIT Fall '26/"
    "MTH 4224 - Intro to Machine Learning/Project1/"
    "lily_project_archive/datasets/v2_original_boxes"
)

# Write a new dataset inside this PyCharm project.
DESTINATION = PROJECT_ROOT / "datasets" / "lily_v1"

if DESTINATION.exists():
    raise FileExistsError(
        f"Destination already exists: {DESTINATION}\n"
        "Choose a new dataset version instead of overwriting it."
    )

# Confirm which class ID means lily.
source_config = yaml.safe_load(
    (SOURCE / "data.yaml").read_text()
)

if source_config["names"][0] != "lily":
    raise ValueError("Expected class 0 to mean lily.")

# Prepare everything in memory before creating output files.
prepared_files = []
summary = {}

for split in ("train", "val"):
    image_folder = SOURCE / "images" / split
    image_paths = sorted(image_folder.glob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No JPG images found in {image_folder}")

    lily_count = 0
    background_count = 0

    for image_path in image_paths:
        label_path = (
            SOURCE / "labels" / split / f"{image_path.stem}.txt"
        )

        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path}")

        lily_labels = []

        for line in label_path.read_text().splitlines():
            fields = line.split()

            if len(fields) != 5:
                raise ValueError(f"Invalid label line: {label_path}")

            # YOLO labels begin with the class ID.
            # Keep lilies (0), excluding anthers (1) and stigmas (2).
            if int(fields[0]) == 0:
                lily_labels.append(line)

        lily_count += len(lily_labels)

        if not lily_labels:
            background_count += 1

        prepared_files.append((split, image_path, lily_labels))

    summary[split] = (
        len(image_paths),
        lily_count,
        background_count,
    )

# Create the new dataset folders.
for split in ("train", "val"):
    (DESTINATION / "images" / split).mkdir(parents=True)
    (DESTINATION / "labels" / split).mkdir(parents=True)

# Copy images and write the filtered annotation files.
for split, image_path, lily_labels in prepared_files:
    shutil.copy2(
        image_path,
        DESTINATION / "images" / split / image_path.name,
    )

    output_label = (
        DESTINATION / "labels" / split / f"{image_path.stem}.txt"
    )

    label_text = "\n".join(lily_labels)

    if label_text:
        label_text += "\n"

    output_label.write_text(label_text)

# Tell YOLO where the images are and which classes exist.
dataset_config = {
    "path": str(DESTINATION),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "lily"},
}

(DESTINATION / "data.yaml").write_text(
    yaml.safe_dump(dataset_config, sort_keys=False)
)

for split, counts in summary.items():
    images, lilies, backgrounds = counts
    print(
        f"{split}: {images} images, {lilies} lily boxes, "
        f"{backgrounds} background-only images"
    )

print(f"Dataset saved to: {DESTINATION}")