"""Place beside heatmap_tools.py in LilyCascade/scripts; run in PyCharm."""

import csv
import hashlib
import html
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from heatmap_tools import (AntherDataset, TinyUNet, find_peaks,
                           heatmap_loss, match_points)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "datasets" / "anther_crops_v1"
RUN_NAME = "anther_unet256_augmented_additionalepochs"
AUGMENT = True

# Change one major setting at a time after examining this first baseline.
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MODEL_WIDTH = 16
FOREGROUND_WEIGHT = 100.0
PEAK_THRESHOLD = 0.30
MIN_PEAK_DISTANCE = 4
MATCH_RADIUS = 6
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def evaluate(model, loader, dataset, device):
    model.eval()
    loss_sum, tp, fp, fn, distances = 0.0, 0, 0, 0, []
    for images, targets, indices in loader:
        logits = model(images.to(device))
        loss_sum += heatmap_loss(logits, targets.to(device),
                                 FOREGROUND_WEIGHT).item() * len(images)
        maps = logits.sigmoid().cpu().numpy()[:, 0]
        for heatmap, index in zip(maps, indices.tolist()):
            peaks = find_peaks(heatmap, PEAK_THRESHOLD, MIN_PEAK_DISTANCE)
            counts = match_points(peaks, dataset.records[index]["points_xy"], MATCH_RADIUS)
            a, b, c, errors = counts
            tp += a
            fp += b
            fn += c
            distances.extend(errors)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"val_loss": loss_sum / len(dataset), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "tp": tp, "fp": fp, "fn": fn,
            "matched_error_px": float(np.mean(distances)) if distances else ""}


@torch.inference_mode()
def save_predictions(model, dataset, device, output):
    """Review every validation crop from the selected best checkpoint."""
    model.eval()
    folder = output / "validation_predictions"
    folder.mkdir()
    cards = []
    for index, record in enumerate(dataset.records):
        image, target, _ = dataset[index]
        heatmap = model(image[None].to(device)).sigmoid().cpu().numpy()[0, 0]
        peaks = find_peaks(heatmap, PEAK_THRESHOLD, MIN_PEAK_DISTANCE)
        picture = image.permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(picture)
        axes[0].set_title("Input crop")
        axes[1].imshow(target[0], cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Annotated target")
        axes[2].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("Predicted heatmap")
        axes[3].imshow(picture)
        if record["points_xy"]:
            x, y = np.asarray(record["points_xy"]).T
            axes[3].scatter(x, y, c="yellow", marker="+", s=60, label="Annotated")
        if peaks:
            x, y, _ = np.asarray(peaks).T
            axes[3].scatter(x, y, edgecolors="cyan", facecolors="none", s=55, label="Predicted")
        if peaks or record["points_xy"]:
            axes[3].legend(fontsize=6)
        axes[3].set_title("Tip locations")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        filename = record["id"] + ".png"
        fig.savefig(folder / filename, dpi=120)
        plt.close(fig)
        cards.append(f'<p>{html.escape(record["id"])}</p>'
                     f'<img loading="lazy" style="max-width:100%" '
                     f'src="validation_predictions/{filename}">')
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>U-Net predictions</title>'
        '<h1>Validation predictions from best checkpoint</h1>' + "".join(cards))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)
    train_data = AntherDataset(DATASET, "train", augment=AUGMENT)
    val_data = AntherDataset(DATASET, "val")
    if train_data.image_size != 256 or val_data.image_size != 256:
        raise ValueError("Pixel-based evaluation settings assume a 256-pixel dataset.")
    output = PROJECT_ROOT / "runs" / RUN_NAME
    if output.exists():
        raise FileExistsError(f"Choose a fresh RUN_NAME; folder exists: {output}")
    output.mkdir(parents=True)
    config = {key: value for key, value in globals().items()
              if key.isupper() and isinstance(value, (str, int, float))}
    config.update(dataset=str(DATASET), device=str(device),
                  architecture="TinyUNet", torch_version=str(torch.__version__),
                  dataset_manifest_sha256=hashlib.sha256(
                      (DATASET / "manifest.json").read_bytes()).hexdigest())
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # Preserve the exact code and manifest used for this experiment.
    for file in (Path(__file__), Path(__file__).with_name("heatmap_tools.py"),
                 DATASET / "manifest.json"):
        (output / file.name).write_bytes(file.read_bytes())
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
    model = TinyUNet(MODEL_WIDTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train crops: {len(train_data)}; validation crops: {len(val_data)}", flush=True)
    print(f"Point scores use threshold {PEAK_THRESHOLD} and matching radius {MATCH_RADIUS} pixels.")
    best_key, stale, history = (-1.0, -float("inf")), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = heatmap_loss(model(images), targets, FOREGROUND_WEIGHT)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        metrics = evaluate(model, val_loader, val_data, device)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_data), **metrics}
        history.append(row)
        with (output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        # Select by point F1; use lower validation loss to break ties.
        key = (metrics["f1"], -metrics["val_loss"])
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "config": config,
                      "metrics": metrics}
        torch.save(checkpoint, output / "last.pt")
        if key > best_key:
            best_key, stale = key, 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        print(f"Epoch {epoch:3d} | loss {row['train_loss']:.5f}/{metrics['val_loss']:.5f}"
              f" | P {metrics['precision']:.3f} R {metrics['recall']:.3f}"
              f" F1 {metrics['f1']:.3f} | TP/FP/FN {metrics['tp']}/{metrics['fp']}/{metrics['fn']}",
              flush=True)
        if stale >= PATIENCE:
            print("Early stopping: validation point F1 has stopped improving.")
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    save_predictions(model, val_data, device, output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("train_loss", "val_loss"):
        axes[0].plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    for name in ("precision", "recall", "f1"):
        axes[1].plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_title("Heatmap loss")
    axes[1].set_title("Point scores on validation crops")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / "results.png", dpi=150)
    plt.close(fig)
    print("Best epoch:", checkpoint["epoch"], "Metrics:", checkpoint["metrics"])
    print("Open:", output / "review.html")


if __name__ == "__main__":
    main()
