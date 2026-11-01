"""
Builds the YOLO-compatible dataset from extracted frames + annotations.

Run this ONCE before starting any training script:
    python prepare_dataset.py

Train/val/test split is done by time (chronological order), NOT randomly,
to avoid temporal autocorrelation bias (dogs lying still for extended periods).

Dataset layout assumed:
    data/extracted_frames/<part>/<image>.jpg
    data/annotation_data/<part>/<image>.txt   (YOLO format)

Output dataset structure (YOLO-compatible):
    data/yolo_dataset/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
    data/yolo_dataset/data.yaml

Both train_yolo_nano.py and train_yolo_medium.py read this same dataset — run
this script only once (or again if the annotations change), NOT once per
training script, otherwise a training run reading the folder while it's
being rebuilt by the other will see a corrupted/partial dataset.
"""

from pathlib import Path
import shutil
import re
import yaml

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(".")
FRAME_DIR        = PROJECT_DIR / "data" / "extracted_frames"
ANNOTATION_DIR   = PROJECT_DIR / "data" / "annotation_data"
YOLO_DATASET_DIR = PROJECT_DIR / "data" / "yolo_dataset"

# ── split boundaries (seconds into the 15-min video) ─────────────────────────
# part_01: 0–300 s  (first 5 min)   → train
# part_02: 300–600 s (middle 5 min) → train + val
# part_03: 600–900 s (last 5 min)   → test
#
# Default: everything before 480 s (8 min) goes to train,
#          480–600 s (8–10 min) goes to val,
#          600 s+ goes to test.
TRAIN_MAX_SEC = 480.0
VAL_MAX_SEC   = 600.0


def parse_time_from_filename(stem: str) -> float:
    """Extract time_sec from filenames like '..._time_0042.50'."""
    match = re.search(r"_time_(\d+\.\d+)$", stem)
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot parse time from filename: {stem}")


def collect_annotated_pairs() -> list[tuple[Path, Path, float]]:
    """
    Return (image_path, label_path, time_sec) for every frame that has
    both an image and a non-empty annotation file, sorted chronologically.
    """
    pairs = []
    for label_path in sorted(ANNOTATION_DIR.glob("**/*.txt")):
        if label_path.stat().st_size == 0:
            continue
        image_path = FRAME_DIR / label_path.parent.name / (label_path.stem + ".jpg")
        if not image_path.exists():
            continue
        time_sec = parse_time_from_filename(label_path.stem)
        pairs.append((image_path, label_path, time_sec))

    pairs.sort(key=lambda x: x[2])
    return pairs


def split_by_time(
    pairs: list[tuple[Path, Path, float]],
    train_max: float,
    val_max: float,
) -> tuple[list, list, list]:
    train, val, test = [], [], []
    for pair in pairs:
        t = pair[2]
        if t < train_max:
            train.append(pair)
        elif t < val_max:
            val.append(pair)
        else:
            test.append(pair)
    return train, val, test


def copy_split(pairs: list[tuple[Path, Path, float]], split_name: str) -> None:
    img_dir = YOLO_DATASET_DIR / "images" / split_name
    lbl_dir = YOLO_DATASET_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path, _ in pairs:
        shutil.copy2(image_path, img_dir / image_path.name)
        shutil.copy2(label_path, lbl_dir / label_path.name)


def create_data_yaml() -> Path:
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    config = {
        "path":  str(YOLO_DATASET_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["dog"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def print_split_summary(train, val, test) -> None:
    total = len(train) + len(val) + len(test)
    print(f"\nDataset split summary (time-based, total annotated: {total})")
    print(f"  train : {len(train):3d} frames  ({len(train)/total*100:.1f}%)  t < {TRAIN_MAX_SEC:.0f}s")
    print(f"  val   : {len(val):3d} frames  ({len(val)/total*100:.1f}%)  {TRAIN_MAX_SEC:.0f}s ≤ t < {VAL_MAX_SEC:.0f}s")
    print(f"  test  : {len(test):3d} frames  ({len(test)/total*100:.1f}%)  t ≥ {VAL_MAX_SEC:.0f}s")
    if len(test) == 0:
        print("  NOTE: test set is empty – part_03 annotations not yet complete.")
    print()


def main():
    pairs = collect_annotated_pairs()
    if not pairs:
        raise RuntimeError("No annotated frames found. Check FRAME_DIR and ANNOTATION_DIR.")

    train, val, test = split_by_time(pairs, TRAIN_MAX_SEC, VAL_MAX_SEC)
    print_split_summary(train, val, test)

    if len(train) == 0:
        raise RuntimeError("Train split is empty.")
    if len(val) == 0:
        raise RuntimeError("Val split is empty. Adjust TRAIN_MAX_SEC / VAL_MAX_SEC.")

    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
    for split_name, split_pairs in [("train", train), ("val", val), ("test", test)]:
        copy_split(split_pairs, split_name)

    yaml_path = create_data_yaml()
    print(f"data.yaml written to: {yaml_path}\n")
    print("Dataset ready. You can now run train_yolo_nano.py and/or train_yolo_medium.py.")


if __name__ == "__main__":
    main()
