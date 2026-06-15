"""
Download QuickDraw data and create train/val/test splits.

Run once before training:
    python data/prepare_data.py

Downloads ~12 MB per class (12 classes = ~144 MB raw).
Saves processed splits to data/processed/.
"""

import json
import numpy as np
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from urllib.parse import quote

CLASSES = [
    "airplane", "banana", "car", "cat", "clock",
    "dog", "donut", "guitar", "ladder", "snowman", "truck", "violin",
]

SAMPLES_PER_CLASS = 5000
SEED              = 42
DOWNLOAD_URL      = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

ROOT          = Path(__file__).parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def download(class_name):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{class_name}.npy"

    if path.exists():
        print(f"  skip  {class_name} (already downloaded)")
        return path

    url = f"{DOWNLOAD_URL}{quote(class_name)}.npy"
    print(f"  download  {class_name}  ←  {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def subsample(path, n):
    arr = np.load(path)              # (N_total, 784) uint8
    idx = np.random.choice(len(arr), size=min(n, len(arr)), replace=False)
    return arr[idx].reshape(-1, 28, 28)   # (n, 28, 28)


def main():
    np.random.seed(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading raw files...")
    paths = {cls: download(cls) for cls in CLASSES}

    print("\nSubsampling and assembling dataset...")
    X_all, y_all = [], []
    class_to_idx = {}

    for i, cls in enumerate(CLASSES):
        X_all.append(subsample(paths[cls], SAMPLES_PER_CLASS))
        y_all.append(np.full(SAMPLES_PER_CLASS, i, dtype=np.int64))
        class_to_idx[cls] = i
        print(f"  {cls}: {SAMPLES_PER_CLASS} samples")

    X = np.concatenate(X_all)    # (60000, 28, 28)
    y = np.concatenate(y_all)    # (60000,)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # 70 / 15 / 15 stratified split
    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val,   y_train, y_val   = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=SEED)

    np.savez_compressed(PROCESSED_DIR / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(PROCESSED_DIR / "val.npz",   X=X_val,   y=y_val)
    np.savez_compressed(PROCESSED_DIR / "test.npz",  X=X_test,  y=y_test)

    with open(PROCESSED_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nDone.")
    print(f"  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    print(f"  saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
