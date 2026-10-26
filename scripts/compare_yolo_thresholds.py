"""Save in LilyCascade/scripts and run. Evaluates validation only, on CPU.

Inference happens once per model. We then filter the saved boxes at different
confidence thresholds: this changes which predictions are accepted, not weights.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.models.yolo.detect.val import DetectionValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MATCH_IOU = 0.45  # Same box-matching tolerance as the matrices we compared.
DEVICE = "cpu"  # Allows the heatmap training job to keep using the Apple GPU.
RUN_NAME = "lily_confidence_comparison"
MODELS = {"416": ("lily_n416_baseline", 416),
          "640": ("lily_n640_baseline", 640)}


def main(project_root=None, output_root=None):
    root = Path(project_root or PROJECT_ROOT)
    dataset = root / "datasets/lily_v1"
    output = Path(output_root or root / "runs" / RUN_NAME)
    if output.exists():
        raise FileExistsError(f"Choose a fresh RUN_NAME: {output}")
    images = sorted(p for p in (dataset / "images/val").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError("No validation images found.")
    rows, predictions = [], {}
    for name, (run, size) in MODELS.items():
        model = YOLO(root / "runs" / run / "weights/best.pt")
        matrices = {t: ConfusionMatrix(names={0: "lily"}) for t in THRESHOLDS}
        cache = []
        # Use the same image preparation and matching as the earlier validations.
        # This subclass records all thresholds during a single validation pass.
        class SweepValidator(DetectionValidator):
            def update_metrics(self, preds, batch):
                for index, pred in enumerate(preds):
                    truth = self._prepare_batch(index, batch)
                    for threshold, matrix in matrices.items():
                        matrix.process_batch(pred, truth, conf=threshold,
                                             iou_thres=MATCH_IOU)
                    cache.append({"image": Path(truth["im_file"]).name,
                                  "coordinate_frame": "resized_padded_validation_input",
                                  "input_hw": list(truth["imgsz"]),
                                  "truth_xyxy": truth["bboxes"].cpu().tolist(),
                                  "boxes_xyxy": pred["bboxes"].cpu().tolist(),
                                  "scores": pred["conf"].cpu().tolist()})
                super().update_metrics(preds, batch)

        model.val(validator=SweepValidator, data=str(dataset / "data.yaml"),
                  imgsz=size, conf=0.001, batch=4, device=DEVICE, workers=0,
                  plots=False, project=str(output), name=f"{name}_validation")
        predictions[name] = cache
        for threshold, matrix in matrices.items():
            tp, fp, fn = int(matrix.matrix[0, 0]), int(matrix.matrix[0, 1]), int(matrix.matrix[1, 0])
            precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append({"model": name, "threshold": threshold, "tp": tp,
                         "fp": fp, "fn": fn, "precision": precision,
                         "recall": recall, "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(
        {"matching_iou": MATCH_IOU, "device": DEVICE,
         "ultralytics_version": ultralytics.__version__, "rows": rows}, indent=2))
    (output / "predictions.json").write_text(json.dumps(predictions))
    lines = ["# Lily confidence comparison", "",
             "Validation only; matching IoU > 0.45. Scores below are not AP.", "",
             "| Model | Confidence | Correct | Extra | Missed | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['threshold']:.2f} | {row['tp']} | "
                     f"{row['fp']} | {row['fn']} | {row['precision']:.1%} | "
                     f"{row['recall']:.1%} | {row['f1']:.1%} |")
    lines += ["", "Choose a threshold according to the false detections your cascade can tolerate.",
              "Maximum F1 is a reference, not automatically the best cascade setting.",
              "The validation set has no background-only photographs; evaluate those on fresh data later.",
              "This measures box detection, not the quality of the resulting anther predictions."]
    (output / "comparison.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name in MODELS:
        selected = [r for r in rows if r["model"] == name]
        for ax, metric in zip(axes, ("precision", "recall", "f1")):
            ax.plot(THRESHOLDS, [r[metric] for r in selected], marker="o", label=name)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Confidence threshold")
            ax.set_ylim(0, 1.02)
            ax.legend(title="Input size")
    fig.tight_layout()
    fig.savefig(output / "threshold_curves.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines))
    print("Saved to:", output)


if __name__ == "__main__":
    main()
