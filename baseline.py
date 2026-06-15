"""
Logistic Regression baseline — classical ML floor for comparison.

Flattens each 28x28 image to 784 pixels and trains a linear classifier.

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES  = [
    "airplane", "banana", "car", "cat", "clock",
    "dog", "donut", "guitar", "ladder", "snowman", "truck", "violin",
]
DATA_DIR = "data/processed"


def load(split):
    data = np.load(f"{DATA_DIR}/{split}.npz")
    X = data["X"].reshape(len(data["X"]), -1) / 255.0   # (N, 784)
    y = data["y"]
    return X, y


print("Loading data...")
X_train, y_train = load("train")
X_test,  y_test  = load("test")

print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples")
print("Training Logistic Regression (this takes ~1 minute)...")

model = LogisticRegression(max_iter=300, random_state=42)
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"\nTest accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm.astype(float) / cm.sum(axis=1, keepdims=True),
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Logistic Regression Baseline  (accuracy = {accuracy:.4f})")
plt.tight_layout()
plt.savefig("results/baseline_confusion_matrix.png", dpi=150)
plt.show()
