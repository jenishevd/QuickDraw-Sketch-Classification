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


## Known Limitations

**Train/serving preprocessing mismatch.** Training data is QuickDraw's canonical 28x28
bitmap, scaled to [0,1] (`train.py`). The demo instead downsizes the raw user canvas
directly to 28x28 with no ink bounding-box crop, centering, or padding (`demo.py`).
A small, off-center sketch loses most of its signal in that resize; a large centered
one doesn't. This is a real distribution shift between training and inference, and
likely compounds the low-confidence/rejection behavior on legitimate but small sketches.
Fix requires defining a canonical crop/pad/centering policy and validating it against
held-out examples — possibly retraining if the new representation differs enough.
Out of scope for this change; flagged, not fixed.

**Guitar/violin confusion (~24-32% misclassification rate).** The confusion matrix
(`evaluate.py`) shows violin→guitar and guitar→violin as the dominant error pair —
violin recall is only ~61%. Not caused by class imbalance (data is stratified,
5,000 samples/class) or an evaluation bug. At 28x28 resolution both classes reduce
to a narrow neck + elongated rounded body; the CNN has no stroke-order or
higher-resolution information to disambiguate them. Fixing this needs richer input
representation (higher resolution or stroke sequence data), targeted hard-example
augmentation, and retraining — a data/modeling change, not a code fix.

## How I used Codex

I used Codex (OpenAI's coding agent) to investigate `evaluate.py` and `demo.py` for
concrete robustness gaps, then to implement a subset of what it found.

**Analysis step:** gave Codex the repo and asked it to identify real, defensible
weak spots — not style nitpicks — across model robustness, input handling, and
evaluation. It returned 5 findings, each with file/line references and a triage
tag (trivial fix / needs design decision / needs retraining).

**What I chose to implement (3 of 5):**
- **Ink-content rejection** — blank canvas, single dots, and near-empty input were
  silently classified instead of rejected. Added a minimum ink-coverage check
  before the tensor conversion.
- **Confidence threshold** — no uncertainty signal existed before returning a
  prediction. Added a top-1 softmax cutoff (0.42, ~5x the 12-class uninformative
  baseline of 1/12) that returns an "uncertain" response instead of a low-confidence guess.
- **Malformed input crash fix** — an empty array or unexpected dimensionality could
  crash `gray.max()` or the PIL conversion. Added validation + try/except before
  processing.

**What I chose not to implement:** the preprocessing mismatch and the guitar/violin
confusion (see Known Limitations above). Both are real issues Codex correctly
identified, but both require a design decision or a retrain to fix properly rather
than a bounded code change — I made the call to document them instead of rushing
a fix I couldn't validate in this timeframe.

**Review process:** each finding was implemented as a separate, scoped commit.
I tested each change manually in the running demo (blank canvas, single dot,
clear sketch, ambiguous guitar/violin sketch, random scribble) rather than
committing on the agent's self-reported test claims alone. One thing I checked
specifically: the ink-check and the existing background-polarity logic
(`gray.mean() > 127`) both reason about foreground/background — I verified they
stayed consistent rather than silently duplicating or contradicting each other.
