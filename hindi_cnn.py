# Author: Rhushil Vasavada
# Devanagari Character CNN — PyTorch port
# Same architecture as devanagari_cnn_modern.py (TF/Keras version), rewritten
# with torchvision datasets/transforms and a standard PyTorch training loop.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- Config ----
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
NUM_CLASSES = 46
EPOCHS = 10
PATIENCE = 2
TRAIN_DIR = "data/DevanagariHandwrittenCharacterDataset/Train"
TEST_DIR = "data/DevanagariHandwrittenCharacterDataset/Test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data ----
# torchvision's ImageFolder expects Train/<class_name>/*.png style layout —
# same structure flow_from_directory / image_dataset_from_directory used.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.2), ratio=(1.0, 1.0)),  # ~zoom_range=0.2
    transforms.ToTensor(),  # scales to [0, 1], equivalent to rescale=1./255
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ---- Model ----
class DevanagariCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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
        # 32x32 -> pool -> 16x16 -> pool -> 8x8, with 64 channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits — softmax handled by loss function


model = DevanagariCNN(NUM_CLASSES).to(device)
print(model)

criterion = nn.CrossEntropyLoss()  # combines log-softmax + NLL, matches "categorical_crossentropy"
optimizer = optim.Adam(model.parameters())

# ---- Training loop with manual early stopping ----
def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(f"Epoch {epoch + 1}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (patience={PATIENCE})")
            break

# restore_best_weights equivalent
if best_state is not None:
    model.load_state_dict(best_state)

# ---- Save ----
# state_dict is the standard, portable way to save PyTorch models
torch.save(model.state_dict(), "hindi_cnn_weights_pytorch.pt")