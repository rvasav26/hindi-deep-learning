from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

NUM_CLASSES = 46

# since PyTorch model outputs English characters representing the Hindi letters, we use
# UTF-8 encoding to represent the corresponding letters in their original Hindi form:
DEVANAGARI_CLASSES: list[str] = [
    "\u091e",
    "\u091f",
    "\u0920",
    "\u0921",
    "\u0922",
    "\u0923",
    "\u0924",
    "\u0925",
    "\u0926",
    "\u0927",
    "\u0915",
    "\u0928",
    "\u092a",
    "\u092b",
    "\u092c",
    "\u092d",
    "\u092e",
    "\u092f",
    "\u0930",
    "\u0932",
    "\u0935",
    "\u0916",
    "\u0936",
    "\u0937",
    "\u0938",
    "\u0939",
    # No UTF-8 encoding for the following Hindi Characters:
    "क्ष",
    "त्र",
    "ज्ञ",
    "\u0917",
    "\u0918",
    "\u0919",
    "\u091a",
    "\u091b",
    "\u091c",
    "\u091d",
    "\u0966",
    "\u0967",
    "\u0968",
    "\u0969",
    "\u096a",
    "\u096b",
    "\u096c",
    "\u096d",
    "\u096e",
    "\u096f",
]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

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
