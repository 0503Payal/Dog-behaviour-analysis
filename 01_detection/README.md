# Detection

Group coursework: frame extraction, manual annotation, and YOLOv8 training.
Built with **Nikolay and Lina**.

## Dataset

240 frames sampled from one 15-minute recording (720×576, 49.95 fps, 2–3 dogs visible),
80 frames from each 5-minute third, annotated by hand in
[makesense.ai](https://www.makesense.ai/) and exported in YOLO format. Single class:
`dog`.

The train/val/test split is **chronological** — 0–8 min train, 8–10 min val, 10–15 min
test — not random. Adjacent frames of the same scene are near-duplicates, so a random
split would leak nearly identical images across the boundary and overstate performance.

## Results

| model | mAP@50 | mAP@50-95 | precision | recall | weights | epochs |
|---|---|---|---|---|---|---|
| nano | 0.972 | 0.537 | 0.939 | 0.948 | 6.2 MB | 80 |
| **nano (tuned)** | 0.981 | **0.606** | 0.989 | 0.958 | **6.2 MB** | 50 |
| medium | **0.992** | **0.608** | 0.990 | **0.967** | 52 MB | 50 |
| large | 0.976 | 0.577 | 0.944 | 0.938 | n/a | 50 |

Medium is the most accurate model. Tuned nano reaches within 0.002 mAP@50-95 of it at an
eighth of the size, which is what makes inference over 45,000 frames practical — so nano
is the deployment choice, not the most accurate one.

**Large is worse than both.** On 240 training images the largest model overfits;
hyperparameter tuning on nano gained more (+0.069 mAP@50-95) than eight times the
parameters did.

> The nano baseline ran 80 epochs against 50 for the others, so the nano-vs-nano-tuned
> comparison mixes tuning with schedule length. A matched 50-epoch nano run is pending.

## Caveat on these numbers

Every frame comes from one recording of one enclosure with the same three dogs. These
metrics describe performance under *this* camera angle and lighting and will not carry
over to a new site without re-annotation.

## Contents

```
scripts/
  prepare_dataset.py       builds the chronological YOLO split
  train_yolo_nano.py       yolov8n
  train_yolo_medium.py     yolov8m
  train_yolo_large.py      yolov8l
  tune_yolo.py             hyperparameter search on nano
notebooks/
  video_inspection_frame_extraction_and_annotation_preparation.ipynb
  evaluate_yolo_comparison.ipynb    nano vs medium vs large
  evaluate_yolo_medium.ipynb
  evaluate_yolo_large.ipynb
results/                   per-run results.csv, args.yaml, curves
```

Model weights and the annotated dataset stay in the original project repository.
