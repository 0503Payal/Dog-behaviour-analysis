"""
YOLOv8-nano training run – small model, fewer epochs, fast.

Target: ~1-2h on CPU. Actual time depends on your machine; watch the
s/it (or it/s) printed for the first couple of epochs in the progress bar
and multiply by EPOCHS to sanity-check the estimate.

Requires data/yolo_dataset to already exist – run prepare_dataset.py first.

Run in its own tmux session, e.g.:
    tmux new -s nano
    conda activate dog_tracking
    python train_yolo_nano.py
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(".")
DATA_YAML   = PROJECT_DIR / "data" / "yolo_dataset" / "data.yaml"

# ── model ────────────────────────────────────────────────────────────────────
# "yolov8n" = Ultralytics' "nano" size (smallest/fastest of their yolov8n/s/m/l/x
# lineup). Auto-downloaded on first run.
PRETRAINED_WEIGHTS = "yolov8n.pt"

# ── training hyperparameters ─────────────────────────────────────────────────
EPOCHS     = 80
IMGSZ      = 640
BATCH_SIZE = 16
DEVICE     = "cpu"

RUN_NAME = "dogs_nano"


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
