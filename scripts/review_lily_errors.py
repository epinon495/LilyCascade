"""Save in LilyCascade/scripts; run to review validation errors in a browser.

Uses the same validation preprocessing and box matching as the threshold sweep.
No training, source annotation edits, or test-set evaluation takes place.
"""
import html
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, box_iou


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = "lily_n416_baseline"
IMAGE_SIZE = 416
CONFIDENCE = 0.10
MATCH_IOU = 0.45
RUN_NAME = "lily_n416_conf010_review"
COLORS = {"annotation": "#25dc63", "prediction": "#169eff",
          "miss": "#ff3030", "extra": "#ff9500"}


def render(image, boxes, path, region=None, size=1000):
    """Draw on a separate preview; retain source images unchanged."""
    region = region or (0, 0, image.width, image.height)
    crop = image.crop(region)
    scale = min(size / crop.width, size / crop.height)
    canvas = crop.resize((max(1, round(crop.width * scale)),
                          max(1, round(crop.height * scale))))
    draw = ImageDraw.Draw(canvas)
    for box, color, label in boxes:
        x0, y0, x1, y1 = box
        mapped = [(x0 - region[0]) * scale, (y0 - region[1]) * scale,
                  (x1 - region[0]) * scale, (y1 - region[1]) * scale]
        draw.rectangle(mapped, outline=color, width=3)
        if label:
            x, y = max(0, mapped[0]), max(0, mapped[1] - 13)
            draw.text((x, y), label, fill=color, stroke_width=1, stroke_fill="black")
    canvas.save(path)


def main(project_root=None, output_root=None):
    root = Path(project_root or PROJECT_ROOT)
    output = Path(output_root or root / "runs" / RUN_NAME)
    if output.exists():
        raise FileExistsError(f"Choose a fresh RUN_NAME: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    records = []

    class ReviewValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            for index, pred in enumerate(preds):
                truth = self._prepare_batch(index, batch)
                matrix = ConfusionMatrix(names={0: "lily"}, save_matches=True)
                matrix.process_batch(pred, truth, conf=CONFIDENCE, iou_thres=MATCH_IOU)

                def original(boxes):
                    if len(boxes) == 0:
                        return []
                    tensor = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.stack(boxes)
                    return ops.scale_boxes(truth["imgsz"], tensor, truth["ori_shape"],
                                           truth["ratio_pad"]).cpu().tolist()

                keep = pred["conf"] > CONFIDENCE
                records.append({"image": str(truth["im_file"]),
                                "annotations": original(truth["bboxes"]),
                                "predictions": original(pred["bboxes"][keep]),
                                "scores": pred["conf"][keep].cpu().tolist(),
                                "misses": original(matrix.matches["FN"]["bboxes"]),
                                "extras": original(matrix.matches["FP"]["bboxes"]),
                                "extra_scores": [float(v) for v in matrix.matches["FP"]["conf"]],
                                "all_boxes": original(pred["bboxes"]),
                                "all_scores": pred["conf"].cpu().tolist(),
                                "tp": int(matrix.matrix[0, 0])})
            super().update_metrics(preds, batch)

    model = YOLO(root / "runs" / MODEL_RUN / "weights/best.pt")
    model.val(validator=ReviewValidator,
              data=str(root / "datasets/lily_v1/data.yaml"), imgsz=IMAGE_SIZE,
              conf=0.001, batch=4, device="cpu", workers=0, plots=False,
              project=str(output), name="validation")
    detail_cards, image_cards = [], []
    for record in records:
        image_path = Path(record["image"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        annotations = [(b, COLORS["annotation"], "") for b in record["annotations"]]
        predictions = [(b, COLORS["prediction"], f"{s:.2f}")
                       for b, s in zip(record["predictions"], record["scores"])]
        missed = [(b, COLORS["miss"], "MISSED") for b in record["misses"]]
        extras = [(b, COLORS["extra"], f"EXTRA {s:.2f}")
                  for b, s in zip(record["extras"], record["extra_scores"])]
        left = f"images/{image_path.stem}_annotations.jpg"
        right = f"images/{image_path.stem}_predictions.jpg"
        render(image, annotations + missed, output / left)
        render(image, predictions + extras, output / right)
        title = html.escape(image_path.name)
        image_cards.append(f'<article><h3>{title}</h3><p>{record["tp"]} matched; '
                           f'{len(missed)} missed; {len(extras)} extra</p>'
                           f'<div class="pair"><a href="{left}"><img loading="lazy" src="{left}"></a>'
                           f'<a href="{right}"><img loading="lazy" src="{right}"></a></div></article>')
        for category, boxes in (("miss", record["misses"]), ("extra", record["extras"])):
            for index, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                margin_x, margin_y = max(60, x1 - x0), max(60, y1 - y0)
                region = (max(0, int(x0 - margin_x)), max(0, int(y0 - margin_y)),
                          min(image.width, int(x1 + margin_x + 1)),
                          min(image.height, int(y1 + margin_y + 1)))
                relative = f"images/{image_path.stem}_{category}_{index:02d}.jpg"
                render(image, annotations + predictions + [(box, COLORS[category], "")],
                       output / relative, region=region, size=500)
                note = ""
                if category == "miss" and record["all_boxes"]:
                    overlaps = box_iou(torch.tensor([box]), torch.tensor(record["all_boxes"]))[0]
                    best = int(overlaps.argmax())
                    note = (f'Highest-overlap prediction (including low scores): '
                            f'IoU {float(overlaps[best]):.2f}, '
                            f'confidence {record["all_scores"][best]:.3f}. '
                            'This is a diagnostic candidate, not a guaranteed identification.')
                elif category == "extra":
                    note = f'Prediction confidence: {record["extra_scores"][index]:.3f}.'
                detail_cards.append(f'<article class="detail" data-kind="{category}">'
                                    f'<h3>{"Missed flower" if category == "miss" else "Extra detection"} '
                                    f'{index + 1}: {title}</h3><p>{note}</p>'
                                    f'<img loading="lazy" src="{relative}"><p>'
                                    f'<a href="{left}">Full annotations</a> · '
                                    f'<a href="{right}">Full predictions</a></p></article>')
    totals = {"matched": sum(r["tp"] for r in records),
              "missed": sum(len(r["misses"]) for r in records),
              "extra": sum(len(r["extras"]) for r in records)}
    report = {"model": MODEL_RUN, "image_size": IMAGE_SIZE, "confidence": CONFIDENCE,
              "matching_iou": MATCH_IOU, "totals": totals, "images": records}
    (output / "review_data.json").write_text(json.dumps(report, indent=2))
    page = '''<!doctype html><meta charset="utf-8"><title>Lily error review</title>
<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#eee}
article{background:white;padding:18px;margin:18px 0;border-radius:10px}img{max-width:100%;max-height:750px}
.pair{display:flex;gap:10px}.pair a{width:50%}button{padding:10px;margin-right:10px}
</style><h1>Lily validation error review</h1>'''
    page += f'<p>YOLO26n / {IMAGE_SIZE} pixels / confidence {CONFIDENCE} / matching IoU &gt; {MATCH_IOU}.</p>'
    page += f'<p><b>{totals["matched"]} matched · {totals["missed"]} missed · {totals["extra"]} extra.</b></p>'
    page += '''<p><b style="color:#c00">Red: missed annotation.</b>
<b style="color:#b66a00">Orange: unmatched prediction.</b>
Green: annotation. Blue: prediction accepted at the selected confidence.</p>
<p>A miss can mean no detection or an insufficiently overlapping box. One poorly sized box
can count as both a miss and an extra detection. Check whether the source annotation itself is correct.</p>
<h2>Close-ups of errors</h2><button onclick="filter('miss')">Missed flowers</button>
<button onclick="filter('extra')">Extra detections</button><button onclick="filter('all')">Both</button>'''
    page += "".join(detail_cards) + '<h2>All validation photographs</h2><p>Left: annotations. Right: predictions.</p>'
    page += "".join(image_cards)
    page += '''<script>function filter(kind){document.querySelectorAll('.detail').forEach(
e=>e.hidden=kind!=='all'&&e.dataset.kind!==kind)}filter('miss')</script>'''
    (output / "review.html").write_text(page)
    print(totals)
    print("Open in your browser:", output / "review.html")


if __name__ == "__main__":
    main()
