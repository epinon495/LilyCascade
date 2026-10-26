from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_FOLDER = (
    PROJECT_ROOT / "datasets" / "annotations" / "anther_endpoints"
)

for filename in ("batch_01.xml", "batch_02.xml", "batch_03.xml"):
    annotation_file = ANNOTATION_FOLDER / filename

    # Parse the CVAT XML document.
    root = ET.parse(annotation_file).getroot()

    # Each image element represents one source photograph.
    images = root.findall("image")

    point_count = 0

    for image in images:
        for annotation in image.findall("points"):
            if annotation.get("label") == "anther_tip":
                coordinates = annotation.get("points")
                point_count += len(coordinates.split(";"))

    print(
        f"{filename}: {len(images)} images, "
        f"{point_count} anther endpoints"
    )