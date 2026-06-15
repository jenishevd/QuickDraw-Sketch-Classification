"""
Train the sketch classifier.

"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import SketchCNN

#  Settings 

CLASSES = [
    "airplane", "banana", "car", "cat", "clock",
    "dog", "donut", "guitar", "ladder", "snowman", "truck", "violin",
]

DATA_DIR   = "data/processed"
EPOCHS     = 15
BATCH_SIZE = 64
LR         = 0.001

# Dataset 

class SketchDataset(Dataset):
    def __init__(self, split):
        data = np.load(f"{DATA_DIR}/{split}.npz")
        # shape: (N, 28, 28) uint8  →  (N, 1, 28, 28) float32 in [0, 1]
        self.X = torch.tensor(data["X"], dtype=torch.float32).unsqueeze(1) / 255.0
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

#  Training 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = DataLoader(SketchDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SketchDataset("val"),   batch_size=BATCH_SIZE)

    model     = SketchCNN(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # --- train ---
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # --- validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images.to(device)).argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)

        val_acc = correct / total
        print(f"Epoch {epoch:2d}/{EPOCHS}  |  val accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "sketch_model.pt")
            print("           →  saved best model")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Weights saved to sketch_model.pt")


if __name__ == "__main__":
    main()
