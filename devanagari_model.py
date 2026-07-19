# Author: Rhushil Vasavada
# Shared model definition, class labels, and loading helper for the Devanagari
# CNN project. Imported by the training script and by both inference apps
# (scratchpad, airpad) so the architecture only lives in one place.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

NUM_CLASSES = 46

# Since the model's output indices correspond to English-labeled folder names
# during training, we map each index to its Devanagari character via UTF-8
# code points for display purposes.
DEVANAGARI_CLASSES: list[str] = [
    u'\u091E', u'\u091F', u'\u0920', u'\u0921', u'\u0922', u'\u0923', u'\u0924', u'\u0925', u'\u0926',
    u'\u0927', u'\u0915', u'\u0928', u'\u092A', u'\u092B', u'\u092c', u'\u092d', u'\u092e', u'\u092f',
    u'\u0930', u'\u0932', u'\u0935', u'\u0916', u'\u0936', u'\u0937', u'\u0938', u'\u0939',
    # No single UTF-8 code point for these three conjunct characters:
    'क्ष', 'त्र', 'ज्ञ',
    u'\u0917', u'\u0918', u'\u0919', u'\u091a', u'\u091b', u'\u091c', u'\u091d', u'\u0966', u'\u0967',
    u'\u0968', u'\u0969', u'\u096a', u'\u096b', u'\u096c', u'\u096d', u'\u096e', u'\u096f',
]

assert len(DEVANAGARI_CLASSES) == NUM_CLASSES


class DevanagariCNN(nn.Module):
    """4-conv-layer CNN for 32x32 grayscale Devanagari character classification."""

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
        return self.classifier(x)  # raw logits — softmax not needed for argmax

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64-dim penultimate-layer activations (pre-final-Linear).

        Useful if you want to inspect or visualize the learned representation
        space rather than just the final class prediction.
        """
        x = self.features(x)
        x = self.classifier[:-1](x)  # everything except the final Linear(64, num_classes)
        return x


def load_trained_model(
    weights_path: str | Path = "hindi_cnn_weights_pytorch.pt",
    device: torch.device | None = None,
) -> tuple[DevanagariCNN, torch.device]:
    """Instantiate DevanagariCNN, load trained weights, and set eval mode."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DevanagariCNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device