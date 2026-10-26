from pathlib import Path
import torch
from ultralytics import YOLO

# Settings

MODEL_NAME = 'yolo26n.pt'
RUN_NAME = 'lily_n640_baseline'

IMAGE_SIZE = 640
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 4

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
MOSAIC = 1.0
SEED = 42

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "datasets" / "lily_v1" / "data.yaml"
PRETRAINED_FOLDER = PROJECT_ROOT / "models" / "pretrained"
RUNS_FOLDER = PROJECT_ROOT / "runs"

MODEL_FILE = PRETRAINED_FOLDER / MODEL_NAME
RUN_FOLDER = RUNS_FOLDER / RUN_NAME

def main():
    # Check the setup before starting a training run.
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if RUN_FOLDER.exists():
        raise FileExistsError(
            f"Run already exists: {RUN_FOLDER}\n"
            "Choose a new RUN_NAME to preserve the previous results."
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError("The Apple GPU is not available.")

    PRETRAINED_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load pretrained weights.
    # Ultralytics downloads this file if it is not already present.
    model = YOLO(str(MODEL_FILE))

    # Fine-tune the model using our lily-only dataset.
    model.train(
        data=str(DATA_FILE),
        project=str(RUNS_FOLDER),
        name=RUN_NAME,
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        mosaic=MOSAIC,
        close_mosaic=10,
        seed=SEED,
        device="mps",
        workers=0,
        plots=True,
        save=True,
    )

    print("\nTraining finished.")
    print("Results folder:", model.trainer.save_dir)
    print("Best checkpoint:", model.trainer.best)


if __name__ == "__main__":
    main()