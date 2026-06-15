# QuickDraw Sketch Classifier

Classifies hand-drawn sketches into 12 categories using a small CNN trained on the Google QuickDraw dataset.

**Classes:** airplane, banana, car, cat, clock, dog, donut, guitar, ladder, snowman, truck, violin

## Setup

```bash
pip install -r requirements.txt
```

## Reproducibility

Model weights (`sketch_model.pt`) are **not included** — run `train.py` to generate them.
Training is fully deterministic (fixed seeds in `train.py`).

## How to run

**Step 1 — Download and prepare data** (run once)
```bash
python data/prepare_data.py
```

**Step 2 — Train the CNN** (produces `sketch_model.pt`)
```bash
python train.py
```

**Step 3 — Evaluate on test set**
```bash
python evaluate.py
```

**Step 4 — Run the interactive demo**
```bash
python demo.py
```

**Optional — Logistic Regression baseline**
```bash
python baseline.py
```

## Project structure

```
model.py          ← CNN architecture (SketchCNN)
train.py          ← training loop (saves sketch_model.pt, not committed)
evaluate.py       ← accuracy, confusion matrix, per-class errors, t-SNE/UMAP
baseline.py       ← Logistic Regression comparison
demo.py           ← Gradio sketchpad demo
data/
  prepare_data.py ← download + split QuickDraw data
  raw/            ← downloaded .npy files
  processed/      ← train/val/test .npz splits
results/          ← saved plots
```

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression (flat pixels) | ~72% |
| SketchCNN | ~87% |
