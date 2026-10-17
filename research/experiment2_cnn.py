# =============================================================================
# experiment2_cnn.py
#
# PURPOSE: Train a small Convolutional Neural Network to classify
#          wildfire vs industrial heat sources from 64×64 thermal patches.
#
# HOW IT WORKS:
#   - The CNN looks at the SPATIAL PATTERN of the patch directly.
#   - Industrial sources have compact, round, intense hot spots.
#   - Wildfires have irregular, spread-out, noisy patterns.
#   - Two convolutional blocks extract these spatial features automatically.
#   - A small fully-connected head produces the final probability.
#
# CNN ARCHITECTURE (deliberately small for CPU training):
#   Conv(1→16, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
#   Conv(16→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
#   Conv(32→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(4×4)
#   FC(64×4×4 → 128) → Dropout(0.3) → FC(128 → 1) → Sigmoid
#
# OUTPUT:
#   results/cnn_scores.npy   — probabilities on the test set
#   results/cnn_labels.npy   — true labels for the test set
#   models/cnn_model.pth     — saved model weights
#   figures/cnn_training.png — loss/accuracy curves
#
# RUN: python experiment2_cnn.py
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED  = 42
TEST_SIZE    = 0.2
BATCH_SIZE   = 128
EPOCHS       = 20       # 20 epochs is enough for a small CNN on CPU
LEARNING_RATE = 1e-3

PATCHES_PATH = "data/patches.npy"
LABELS_PATH  = "data/patch_labels.npy"

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading patches …")

if not os.path.exists(PATCHES_PATH):
    raise FileNotFoundError(
        f"{PATCHES_PATH} not found. Run step2_extract_patches.py first."
    )

patches = np.load(PATCHES_PATH, mmap_mode="r")   # shape (N, 64, 64)
labels  = np.load(LABELS_PATH)    # shape (N,)

print(f"  Patches: {patches.shape}  Labels: {labels.shape}")
print(f"  Wildfire (0): {(labels==0).sum():,}   Industrial (1): {(labels==1).sum():,}")


# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
indices = np.arange(len(labels))
train_indices, test_indices, y_train, y_test = train_test_split(
    indices, labels, test_size=TEST_SIZE, stratify=labels,
    random_state=RANDOM_SEED
)
print(f"\nTrain: {len(train_indices):,}   Test: {len(test_indices):,}")


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset and DataLoader
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Read patches from disk-backed storage as batches request them."""

    def __init__(self, patches_np, labels_np, indices, augment=False):
        self.patches = patches_np
        self.labels = labels_np
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patch_idx = self.indices[idx]
        patch = np.array(self.patches[patch_idx], copy=True)
        x = torch.from_numpy(patch).unsqueeze(0)
        y = torch.tensor(self.labels[patch_idx], dtype=torch.float32)

        if self.augment:
            # Simple augmentations: random horizontal/vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])   # horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])   # vertical flip
            # Random 90-degree rotations
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])

        return x, y


# Class weights to handle imbalance
n_wild = (y_train == 0).sum()
n_ind  = (y_train == 1).sum()
# Weight for positive (industrial) class
pos_weight = torch.tensor([n_wild / max(n_ind, 1)], dtype=torch.float32)
print(f"\nClass imbalance ratio (neg/pos): {pos_weight.item():.2f}")

train_dataset = PatchDataset(patches, labels, train_indices, augment=True)
test_dataset  = PatchDataset(patches, labels, test_indices, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# 4. CNN Model
# ---------------------------------------------------------------------------

class ThermalCNN(nn.Module):
    """
    Small CNN for 64×64 single-channel thermal patches.
    Designed to train in minutes on CPU.
    """

    def __init__(self):
        super().__init__()

        # Block 1: detect edges and local hotspots
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 64×64 → 64×64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                               # 64×64 → 32×32
        )

        # Block 2: detect medium-scale patterns (compact vs spread)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32×32 → 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                # 32×32 → 16×16
        )

        # Block 3: detect global shape signature
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16×16 → 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)                        # → 4×4 regardless of input size
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),                        # 64 × 4 × 4 = 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)                    # raw logit
            # No sigmoid here — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)   # shape (batch,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = ThermalCNN().to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# BCEWithLogitsLoss = Sigmoid + BCELoss (numerically stable)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reduce learning rate if val loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
print(f"\nTraining for {EPOCHS} epochs …\n")

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

for epoch in range(1, EPOCHS + 1):

    # ---- Training phase ----
    model.train()
    batch_losses, batch_correct, batch_total = [], 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        preds = (torch.sigmoid(logits) >= 0.5).long()
        batch_correct += (preds == y_batch.long()).sum().item()
        batch_total   += len(y_batch)

    train_losses.append(np.mean(batch_losses))
    train_accs.append(100 * batch_correct / batch_total)

    # ---- Validation phase ----
    model.eval()
    val_batch_losses, val_correct, val_total = [], 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            val_batch_losses.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).long()
            val_correct += (preds == y_batch.long()).sum().item()
            val_total   += len(y_batch)

    val_losses.append(np.mean(val_batch_losses))
    val_accs.append(100 * val_correct / val_total)
    scheduler.step(val_losses[-1])

    print(f"  Epoch {epoch:>3}/{EPOCHS}  "
          f"train loss: {train_losses[-1]:.4f}  acc: {train_accs[-1]:.1f}%  |  "
          f"val loss: {val_losses[-1]:.4f}  acc: {val_accs[-1]:.1f}%")


# ---------------------------------------------------------------------------
# 6. Evaluate on test set
# ---------------------------------------------------------------------------
print("\nEvaluating on test set …")

model.eval()
all_proba, all_true = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        proba   = torch.sigmoid(logits).cpu().numpy()
        all_proba.append(proba)
        all_true.append(y_batch.numpy())

y_proba = np.concatenate(all_proba)
y_true  = np.concatenate(all_true).astype(int)
y_pred  = (y_proba >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
                             target_names=["wildfire", "industrial"]))

auc = roc_auc_score(y_true, y_proba)
print(f"ROC-AUC: {auc:.4f}")


# ---------------------------------------------------------------------------
# 7. Save results and model
# ---------------------------------------------------------------------------
np.save("results/cnn_scores.npy", y_proba)
np.save("results/cnn_labels.npy", y_true)
torch.save(model.state_dict(), "models/cnn_model.pth")
print("\nSaved: results/cnn_scores.npy  results/cnn_labels.npy  models/cnn_model.pth")


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Experiment 2 — CNN Training", fontsize=14, fontweight="bold")

# Loss curves
ax = axes[0]
ax.plot(train_losses, label="Train loss", color="steelblue")
ax.plot(val_losses,   label="Val loss",   color="orange")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
ax.set_title("Loss Curves"); ax.legend(); ax.grid(alpha=0.3)

# Accuracy curves
ax = axes[1]
ax.plot(train_accs, label="Train acc", color="steelblue")
ax.plot(val_accs,   label="Val acc",   color="orange")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Curves"); ax.legend(); ax.grid(alpha=0.3)

# Confusion matrix
ax = axes[2]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["wildfire", "industrial"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"CNN Confusion Matrix  (AUC = {auc:.3f})")

plt.tight_layout()
plt.savefig("figures/cnn_training.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/cnn_training.png")

print("\nExperiment 2 complete.")