# ============================================================
# SimCLR training script (Slurm-friendly, no tqdm)
# ============================================================

# -------------------------
# Configuration
# -------------------------
SEED = 666
DEVICE = "cuda:0"

DATASET = "PCAM"  # "PCAM" or "STL10"
DATA_DIR = "Week4/data"
IMG_SIZE = 64
UNLABELED_SUBSET = None  # e.g. 5000 for STL10, None = full

BATCH_SIZE = 8
EPOCHS = 10
LR = 3e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
USE_AMP = True

NUM_WORKERS = 1
DROP_LAST = True

SAVE_DIR = "Week4/models"
RUN_NAME = f"resnet18_simclr_{EPOCHS}epochs_{DATASET}.pth.tar"

# -------------------------
# Imports & setup
# -------------------------
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import PCAM, STL10

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(DEVICE)

print("=== SimCLR Training ===")
print(f"Dataset: {DATASET}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"Device: {device}")
print()


# -------------------------
# Data augmentations
# -------------------------
class ContrastiveTransformations:
    def __init__(self, img_size, s=1.0):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * img_size) // 2 * 2 + 1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# -------------------------
# Dataset
# -------------------------
if DATASET == "PCAM":
    unlabeled_dataset = PCAM(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=ContrastiveTransformations(IMG_SIZE),
    )

elif DATASET == "STL10":
    full_unlabeled = STL10(
        root=DATA_DIR,
        split="unlabeled",
        download=True,
        transform=ContrastiveTransformations(96),
    )

    if UNLABELED_SUBSET is not None:
        unlabeled_dataset, _ = random_split(
            full_unlabeled,
            [UNLABELED_SUBSET, len(full_unlabeled) - UNLABELED_SUBSET],
        )
    else:
        unlabeled_dataset = full_unlabeled
else:
    raise ValueError("Unknown dataset")

print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")

loader = DataLoader(
    unlabeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=DROP_LAST,
    pin_memory=True,
)


# -------------------------
# Model
# -------------------------
def build_simclr_resnet18(out_dim=128):
    backbone = models.resnet18(weights=None)
    dim = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, out_dim),
    )
    return backbone


model = build_simclr_resnet18().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=len(loader),
)

criterion = nn.CrossEntropyLoss()
scaler = GradScaler(enabled=USE_AMP)

# -------------------------
# Training
# -------------------------
print("Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for images, _ in loader:
        images = torch.cat(images, dim=0).to(device)

        with torch.autocast("cuda", enabled=USE_AMP):
            feats = F.normalize(model(images), dim=1)

            labels = torch.cat([torch.arange(BATCH_SIZE) for _ in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

            sim = feats @ feats.T
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

            labels = labels[~mask].view(labels.shape[0], -1)
            sim = sim[~mask].view(sim.shape[0], -1)

            positives = sim[labels.bool()].view(labels.shape[0], -1)
            negatives = sim[~labels.bool()].view(sim.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1) / TEMPERATURE
            targets = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            loss = criterion(logits, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"loss = {epoch_loss / n_batches:.4f} | "
        f"lr = {scheduler.get_last_lr()[0]:.6f}"
    )

# -------------------------
# Save checkpoint
# -------------------------
ckpt_path = os.path.join(SAVE_DIR, RUN_NAME)
torch.save(
    {"epoch": EPOCHS, "state_dict": model.state_dict()},
    ckpt_path,
)

print("\nTraining finished.")
print(f"Model saved to: {ckpt_path}")
