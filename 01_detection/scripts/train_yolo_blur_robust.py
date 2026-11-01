"""
Retrain the detector to survive motion blur.

Diagnostics showed detection confidence falling from 0.90 mid-track to ~0.72
where tracks break, with bounding boxes nearly twice as large in the fastest
speed tercile. That is motion blur: at 720x576 and 49.95 fps, a fast dog smears
across the frame and the detector loses confidence in it. Because the loss is
concentrated on fast movement, it biases every behavioural statistic.

Two changes address it directly:

  1. Train at a larger input size. The default 640 downscales a 720x576 frame,
     throwing away detail exactly where the blurred edges are.

  2. Add motion blur to the training data, so the model sees blurred dogs as
     dogs. Ultralytics applies only mild blur via albumentations by default
     (p=0.01); this generates directional motion blur, which is what a moving
     animal actually produces, and adds it as extra training images.

Run `python train_yolo_blur_robust.py --make-blur` once to build the augmented
copies, then run without the flag to train.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATASET = Path("data/yolo_dataset")
BLUR_KERNELS = (9, 15, 21)     # increasing severity, in pixels of smear
ANGLES_DEG = (0, 45, 90, 135)


def motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    """A line-shaped kernel: convolving with it smears along `angle_deg`."""
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    centre = (size / 2 - 0.5, size / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    return kernel / kernel.sum()


def build_blurred_copies(split: str = "train") -> int:
    """Write a motion-blurred duplicate of each training image, labels reused.

    Blur does not move a bounding box, so the original label file is correct
    for the blurred copy and is simply copied alongside it.
    """
    images = DATASET / "images" / split
    labels = DATASET / "labels" / split
    written = 0

    for image_path in sorted(images.glob("*.jpg")):
        if "_blur" in image_path.stem:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        size = BLUR_KERNELS[written % len(BLUR_KERNELS)]
        angle = ANGLES_DEG[written % len(ANGLES_DEG)]
        blurred = cv2.filter2D(image, -1, motion_blur_kernel(size, angle))

        out_stem = f"{image_path.stem}_blur{size}"
        cv2.imwrite(str(images / f"{out_stem}.jpg"), blurred)

        label_path = labels / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, labels / f"{out_stem}.txt")
        written += 1

    print(f"wrote {written} motion-blurred training images")
    return written


def train(imgsz: int, epochs: int, device: str) -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATASET / "data.yaml"),
        imgsz=imgsz,       # 960 instead of 640: keep the detail blur destroys
        epochs=epochs,
        batch=8,           # larger imgsz needs a smaller batch
        device=device,
        name="dogs_nano_blur_robust",
        patience=20,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-blur", action="store_true",
                        help="generate blurred training copies, then exit")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0", help='"0" for GPU, "cpu" otherwise')
    args = parser.parse_args()

    if args.make_blur:
        build_blurred_copies()
    else:
        train(args.imgsz, args.epochs, args.device)
