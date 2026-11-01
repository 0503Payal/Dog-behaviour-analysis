"""
Hyperparameter tuning for the dog-detection YOLO project.
 
Uses Ultralytics' built-in evolutionary tuner (model.tune):
- trains the model `ITERATIONS` times with mutated hyperparameters
- selects on VALIDATION fitness (test split stays untouched)
- writes results to runs/detect/tune/:
    best_hyperparameters.yaml  <- the winning config
    tune_results.csv           <- fitness per iteration
    tune_scatter_plots.png     <- which params mattered
 
Afterwards, retrain once with the best hyperparameters (see bottom),
and only THEN evaluate on the test split.
"""
 
from pathlib import Path
from ultralytics import YOLO
 
PROJECT_DIR = Path(".")
DATA_YAML   = PROJECT_DIR / "data" / "yolo_dataset" / "data.yaml"
 
# Tune the SMALL model: iterations are cheap, and the nano-vs-medium
# comparison already showed capacity is not the bottleneck on 240 frames.
PRETRAINED_WEIGHTS = "yolov8n.pt"
 
EPOCHS_PER_ITER = 30    # medium peaked at epoch ~14, so 30 is plenty
ITERATIONS      = 30    # raise to 100+ on a GPU; lower to ~15 on CPU if needed
DEVICE          = "mps" # set "0" on the GPU cluster or "cpu" on a CPU-only machine
 
# Restrict the search space to what plausibly matters for a small,
# overfitting-prone dataset: learning rate, regularization, augmentation.
# Format: param: (min, max)
SEARCH_SPACE = {
    "lr0":          (1e-4, 3e-3),   # AdamW-appropriate range
    "lrf":          (0.01, 0.1),    # final LR fraction (cosine schedule end)
    "weight_decay": (0.0001, 0.001),# regularization strength
    "hsv_h":        (0.0, 0.1),    # color augmentation
    "hsv_s":        (0.0, 0.9),
    "hsv_v":        (0.0, 0.9),
    "degrees":      (0.0, 45.0),    # rotation
    "translate":    (0.0, 0.9),
    "scale":        (0.0, 0.95),     # scale jitter
    "mosaic":       (0.0, 1.0),
    "mixup":        (0.0, 1.0),     # extra regularization via image mixing
    "fliplr":       (0.0, 1.0),
}
 
def main():
    assert DATA_YAML.exists(), f"{DATA_YAML} not found - run prepare_dataset.py first."
 
    model = YOLO(PRETRAINED_WEIGHTS)
 
    model.tune(
        data=str(DATA_YAML),
        epochs=EPOCHS_PER_ITER,
        iterations=ITERATIONS,
        space=SEARCH_SPACE,
        optimizer="AdamW",
        device=DEVICE,
        imgsz=640,
        batch=16,          # fixed: don't tune batch on CPU, AutoBatch needs GPU
        patience=10,       # early-stop iterations that stop improving
        plots=False,       # keep iterations lean
        val=True,          # fitness = validation metrics (test stays clean!)
        name="tune_nano",
    )
 
    print("\nDone. Best hyperparameters written to:")
    print("  runs/detect/tune_nano/best_hyperparameters.yaml")
    print("\nNext step - retrain ONCE with the winners, e.g.:")
    print("  model = YOLO('yolov8n.pt')")
    print("  model.train(data=..., epochs=50, cfg='runs/detect/tune_nano/best_hyperparameters.yaml')")

def train():

    # ── Stage 2: final training with the winners ──────────────────────────
    best_hp_yaml = PROJECT_DIR / "runs" / "detect" / "tune_nano" / "best_hyperparameters.yaml"
    assert best_hp_yaml.exists(), f"{best_hp_yaml} not found - did tuning fail?"

    final_model = YOLO(PRETRAINED_WEIGHTS)          # fresh start from COCO weights
    final_model.train(
        data=str(DATA_YAML),
        cfg=str(best_hp_yaml),      # apply tuned hyperparameters
        epochs=50,
        patience=15,
        device=DEVICE,
        imgsz=640,
        batch=16,
        name="dogs_nano_tuned",     # -> runs/detect/dogs_nano_tuned/
    )
    return final_model
    # after train(), final_model automatically holds best.pt of this run
    
def test(final_model):
    # # ── Stage 3: the one and only test evaluation ─────────────────────────
    print("\nEvaluating on test set ...")
    metrics = final_model.val(data=str(DATA_YAML), split="test", device=DEVICE, name="tune_nano_test")
    print(f"Test  mAP@50: {metrics.box.map50:.4f}  mAP@50-95: {metrics.box.map:.4f}")
    print(f"Test  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
 
if __name__ == "__main__":
    final_model = train()
    test(final_model)