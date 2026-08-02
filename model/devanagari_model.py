from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

NUM_CLASSES = 46


class DevanagariCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        embedding = self.classifier[:-1](x)
        logits = self.classifier[-1](embedding)

        return logits, embedding

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier[:-1](x)
        return x


def load_trained_model(
    weights_path: str | Path = "hindi_cnn_weights_pytorch.pt",
    device: torch.device | None = None,
) -> tuple[DevanagariCNN, torch.device]:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DevanagariCNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device
