"""
YOLOv8-large training run – Optimized for high-performance computing.
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(".")
DATA_YAML   = PROJECT_DIR / "data" / "yolo_dataset" / "data.yaml"

# ── model ────────────────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = "yolov8l.pt"

# ── optimized training hyperparameters ───────────────────────────────────────
EPOCHS     = 50
IMGSZ      = 640
BATCH_SIZE = -1       # AutoBatch: automatically maxes out batch size for your GPU
WORKERS    = 8        # Increased to prepare images much faster
CACHE      = True     # Loads images into RAM, eliminating read/write bottlenecks
DEVICE     = "cpu"        # "0" uses the first GPU. (Change to "cpu" ONLY if no GPU is available)

RUN_NAME = "dogs_large"

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