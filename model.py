import torch.nn as nn


class SketchCNN(nn.Module):
    """
    Simple CNN for classifying 28x28 grayscale sketches.

    Architecture:
        Conv block 1: 1 -> 32 channels, 28x28 -> 14x14
        Conv block 2: 32 -> 64 channels, 14x14 -> 7x7
        Classifier:   flatten -> 256 -> num_classes
    """

    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28x28 -> 14x14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),             # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))
