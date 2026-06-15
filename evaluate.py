"""
Evaluate the trained model on the test set.

Produces:
  results/confusion_matrix.png   — normalised confusion matrix
  results/per_class_errors.png   — per-class error breakdown (top confusions)
  results/tsne_embeddings.png    — t-SNE of penultimate-layer features
  results/umap_embeddings.png    — UMAP of penultimate-layer features

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from train import SketchDataset, CLASSES
from model import SketchCNN

# Load model and data

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataLoader(SketchDataset("test"), batch_size=64)

model = SketchCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("sketch_model.pt", map_location=device))
model.eval()

# Collect predictions + penultimate-layer features 
# Hook extracts the 256-dim ReLU activations (before the final linear layer).

features_list = []

def _hook(module, input, output):
    features_list.append(output.detach().cpu())

# classifier = [Flatten, Linear(3136→256), ReLU, Dropout, Linear(256→C)]
hook_handle = model.classifier[2].register_forward_hook(_hook)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

hook_handle.remove()

y_true    = np.array(y_true)
y_pred    = np.array(y_pred)
features  = torch.cat(features_list).numpy()   # shape: (N, 256)

# Print results

accuracy = (y_true == y_pred).mean()
print(f"Test accuracy: {accuracy:.4f}\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion matrix

cm            = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix  (test accuracy = {accuracy:.4f})")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("Saved: results/confusion_matrix.png")

# Per-class error analysis 
# For each class, show the top-3 misclassification targets.

TOP_N = 3
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
axes = axes.flatten()

for i, cls in enumerate(CLASSES):
    mask      = y_true == i
    errors    = y_pred[mask & (y_pred != i)]  # wrong predictions only
    error_pct = len(errors) / mask.sum() * 100

    if len(errors) == 0:
        axes[i].set_title(f"{cls}\n(0 errors)")
        axes[i].axis("off")
        continue

    targets, counts = np.unique(errors, return_counts=True)
    order           = np.argsort(counts)[::-1][:TOP_N]
    top_targets     = [CLASSES[t] for t in targets[order]]
    top_counts      = counts[order] / mask.sum() * 100   # as % of class total

    axes[i].barh(top_targets[::-1], top_counts[::-1], color="salmon")
    axes[i].set_xlabel("% of class samples")
    axes[i].set_title(f"{cls}  (error rate {error_pct:.1f}%)")
    axes[i].set_xlim(0, max(top_counts) * 1.3)

    for j, (v, lbl) in enumerate(zip(top_counts[::-1], top_targets[::-1])):
        axes[i].text(v + 0.3, j, f"{v:.1f}%", va="center", fontsize=8)

fig.suptitle("Per-class Top Confusions", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/per_class_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/per_class_errors.png")

# t-SNE of penultimate-layer embeddings 

print("Running t-SNE  (this takes ~30s) …")
tsne  = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
emb2d = tsne.fit_transform(features)

plt.figure(figsize=(9, 7))
palette = plt.get_cmap("tab20", len(CLASSES))
for i, cls in enumerate(CLASSES):
    mask = y_true == i
    plt.scatter(emb2d[mask, 0], emb2d[mask, 1],
                s=6, alpha=0.6, color=palette(i), label=cls)
plt.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.title("t-SNE — penultimate-layer embeddings (256-d → 2-d)")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/tsne_embeddings.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/tsne_embeddings.png")

# UMAP of penultimate-layer embeddings 

try:
    import umap
    print("Running UMAP …")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        random_state=42)
    emb_umap = reducer.fit_transform(features)

    plt.figure(figsize=(9, 7))
    for i, cls in enumerate(CLASSES):
        mask = y_true == i
        plt.scatter(emb_umap[mask, 0], emb_umap[mask, 1],
                    s=6, alpha=0.6, color=palette(i), label=cls)
    plt.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.title("UMAP — penultimate-layer embeddings (256-d → 2-d)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/umap_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/umap_embeddings.png")
except ImportError:
    print("umap-learn not installed — skipping UMAP  (pip install umap-learn)")
