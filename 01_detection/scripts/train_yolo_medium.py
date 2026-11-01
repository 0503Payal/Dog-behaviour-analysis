"""
YOLOv8-medium training run – bigger model, more epochs, slower.

Target: ~3-4h on CPU for 50 epochs. Actual time depends on your machine; watch the
s/it (or it/s) printed for the first couple of epochs in the progress bar
and multiply by EPOCHS to sanity-check the estimate. If the first epochs
suggest it'll badly overshoot that, lower EPOCHS further (edit the file, Ctrl+C
the run, and restart it – YOLO also supports resuming, see 'resume=True').

Requires data/yolo_dataset to already exist – run prepare_dataset.py first.

Run in its own tmux session, e.g.:
    tmux new -s medium
    conda activate dog_tracking
    python train_yolo_medium.py
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(".")
DATA_YAML   = PROJECT_DIR / "data" / "yolo_dataset" / "data.yaml"

# ── model ────────────────────────────────────────────────────────────────────
# "yolov8m" = Ultralytics' "medium" size (bigger/slower than the "nano" size used by
# train_yolo_nano.py). Auto-downloaded on first run.
PRETRAINED_WEIGHTS = "yolov8m.pt"

# ── training hyperparameters ─────────────────────────────────────────────────
EPOCHS     = 50
IMGSZ      = 640
BATCH_SIZE = 4    # reduced from 8 – previous run got OOM-killed under WSL2's ~7GB RAM limit
WORKERS    = 2    # fewer dataloader workers = fewer parallel image/mosaic buffers in RAM
CACHE      = False  # don't cache images in RAM
DEVICE     = "cpu"

RUN_NAME = "dogs_medium"


def main():
    assert DATA_YAML.exists(), (
        f"{DATA_YAML} not found – run 'python prepare_dataset.py' first."
    )

    model = YOLO(PRETRAINED_WEIGHTS)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=False,
        save_period=10,
        # augmentation – helpful for small datasets
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        degrees=5.0,
    )

    print("\nEvaluating on test set ...")
    model.val(data=str(DATA_YAML), split="test")


if __name__ == "__main__":
    main()
