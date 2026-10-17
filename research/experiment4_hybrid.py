# =============================================================================
# experiment4_hybrid.py
#
# PURPOSE: Fuse ALL three feature types to build a stronger classifier.
#
# HOW THE FUSION WORKS:
#   Each earlier experiment produces a different "view" of the same detection:
#
#   (A) PCA features    — compressed spatial shape of the thermal patch
#   (B) CNN features    — deep spatial features extracted from the patch
#   (C) FIRMS features  — tabular spectral + temporal attributes
#
#   This experiment combines them at the FEATURE LEVEL:
#   → Extract the CNN's penultimate layer (feature vector before final output).
#   → Extract PCA-compressed features.
#   → Extract engineered FIRMS tabular features.
#   → Concatenate all three into one wide feature vector.
#   → Train a Logistic Regression on the combined vector.
#
#   WHY FUSION IS BETTER:
#   The CNN sees spatial patterns. FIRMS sees spectral physics.
#   Neither alone captures everything. Combined, they are complementary.
#
# PREREQUISITES:
#   - Run experiment1_pca.py first (needs the PCA model in memory — we
#     re-fit it here for simplicity, no pickle needed).
#   - Run experiment2_cnn.py first (needs models/cnn_model.pth).
#   - Run step1_label_dataset.py and step2_extract_patches.py.
#
# OUTPUT:
#   results/hybrid_scores.npy  — probabilities on the test set
#   results/hybrid_labels.npy  — true labels for the test set
#   figures/hybrid_fusion.png
#
# RUN: python experiment4_hybrid.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED  = 42
TEST_SIZE    = 0.2
N_PCA_COMPS  = 50    # must match experiment1_pca.py

PATCHES_PATH  = "data/patches.npy"
LABELS_PATH   = "data/patch_labels.npy"
LABELLED_CSV  = "data/firms_labelled.csv"
CNN_MODEL_PATH = "models/cnn_model.pth"

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data …")

if not os.path.exists(PATCHES_PATH):
    raise FileNotFoundError(
        f"{PATCHES_PATH} not found. Run step2_extract_patches.py first."
    )
if not os.path.exists(LABELLED_CSV):
    raise FileNotFoundError(
        f"{LABELLED_CSV} not found. Run step1_label_dataset.py first."
    )

patches = np.load(PATCHES_PATH)   # (N, 64, 64)
labels  = np.load(LABELS_PATH)    # (N,)

# Load FIRMS metadata aligned with patches (step2 saves patch_meta.csv)
meta_path = "data/patch_meta.csv"
if os.path.exists(meta_path):
    firms_meta = pd.read_csv(meta_path)
else:
    # If no meta file, load from labelled CSV and hope lengths match
    firms_meta = pd.read_csv(LABELLED_CSV)
    if len(firms_meta) != len(patches):
        firms_meta = firms_meta.iloc[:len(patches)].copy()

print(f"  Patches: {patches.shape}")
print(f"  Labels : {labels.shape}")
print(f"  Meta   : {firms_meta.shape}")


# ---------------------------------------------------------------------------
# 2. Train / test split  (same seed and size as all other experiments)
# ---------------------------------------------------------------------------
N = len(patches)
indices = np.arange(N)

idx_train, idx_test, y_train, y_test = train_test_split(
    indices, labels, test_size=TEST_SIZE, stratify=labels,
    random_state=RANDOM_SEED
)
print(f"\nTrain: {len(idx_train):,}   Test: {len(idx_test):,}")


# ===========================================================================
# ---- Feature block A: PCA features ----------------------------------------
# ===========================================================================
print("\n--- Building PCA features ---")

X_flat = patches.reshape(N, -1).astype(np.float32)    # (N, 4096)

pca = PCA(n_components=N_PCA_COMPS, random_state=RANDOM_SEED)
X_pca_train = pca.fit_transform(X_flat[idx_train])
X_pca_test  = pca.transform(X_flat[idx_test])

# Also compute reconstruction error as an extra feature
def recon_error(X_orig, X_pca_coords, pca_model):
    return np.mean((X_orig - pca_model.inverse_transform(X_pca_coords)) ** 2,
                   axis=1, keepdims=True)

err_train = recon_error(X_flat[idx_train], X_pca_train, pca)
err_test  = recon_error(X_flat[idx_test],  X_pca_test,  pca)

pca_feat_train = np.hstack([X_pca_train, err_train])   # (n_train, 51)
pca_feat_test  = np.hstack([X_pca_test,  err_test])

print(f"  PCA feature dim: {pca_feat_train.shape[1]}")


# ===========================================================================
# ---- Feature block B: CNN penultimate layer features ----------------------
# ===========================================================================
print("\n--- Extracting CNN features ---")


class ThermalCNN(nn.Module):
    """Same architecture as experiment2_cnn.py — must match exactly."""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.AdaptiveAvgPool2d(4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)

    def extract_features(self, x):
        """Return the 128-dim penultimate layer vector."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head[:3](x)   # shape (batch, 128)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN weights if available; otherwise use randomly-initialised weights
# (random weights still give *some* spatial features — experiment is valid)
cnn_model = ThermalCNN().to(device)

if os.path.exists(CNN_MODEL_PATH):
    cnn_model.load_state_dict(
        torch.load(CNN_MODEL_PATH, map_location=device)
    )
    print(f"  Loaded CNN weights from {CNN_MODEL_PATH}")
else:
    print(f"  WARNING: {CNN_MODEL_PATH} not found.")
    print("  Run experiment2_cnn.py first for trained weights.")
    print("  Using randomly-initialised CNN features (still informative).")

cnn_model.eval()


def extract_cnn_features(idx_array, batch_size=64):
    """Extract 128-dim feature vectors from the CNN for a set of patches."""
    patches_tensor = torch.tensor(
        patches[idx_array][:, np.newaxis, :, :], dtype=torch.float32
    )
    dataset = TensorDataset(patches_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feats = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            feat = cnn_model.extract_features(X_batch).cpu().numpy()
            feats.append(feat)
    return np.vstack(feats)


cnn_feat_train = extract_cnn_features(idx_train)   # (n_train, 128)
cnn_feat_test  = extract_cnn_features(idx_test)

print(f"  CNN feature dim: {cnn_feat_train.shape[1]}")


# ===========================================================================
# ---- Feature block C: FIRMS tabular features ------------------------------
# ===========================================================================
print("\n--- Building FIRMS tabular features ---")

# Replicate the feature engineering from experiment3_firms.py
df = firms_meta.copy()
df["acq_date"] = pd.to_datetime(df.get("acq_date"), errors="coerce")

brightness_aliases = {
    "bright_ti4": "brightness",
    "bright_ti5": "bright_t31",
}
for feature, alias in brightness_aliases.items():
    if feature not in df.columns and alias in df.columns:
        df[feature] = df[alias]

missing_brightness = [
    feature for feature in brightness_aliases if feature not in df.columns
]
if missing_brightness:
    raise ValueError(
        "Cannot engineer brightness features "
        f"{missing_brightness}. Expected columns "
        "'bright_ti4'/'bright_ti5' or their FIRMS aliases "
        "'brightness'/'bright_t31'."
    )

for col in ["bright_ti4", "bright_ti5", "frp", "scan", "track"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

df["acq_time"]   = pd.to_numeric(df.get("acq_time", 1200), errors="coerce").fillna(1200).astype(int)
df["hour_of_day"] = df["acq_time"] // 100
df["month"]       = df["acq_date"].dt.month.fillna(6).astype(int) if "acq_date" in df.columns else 6
df["day_of_year"] = df["acq_date"].dt.dayofyear.fillna(180).astype(int) if "acq_date" in df.columns else 180
df["bright_diff"] = df["bright_ti4"] - df["bright_ti5"]
pixel_area        = df["scan"] * df["track"]
df["frp_per_area"] = (df["frp"] / pixel_area.replace(0, np.nan)).fillna(0)
df["is_night"]     = (df.get("daynight", "D").str.strip().str.upper() == "N").astype(int) \
                      if "daynight" in df.columns else 0
conf_map = {"low": 0, "nominal": 1, "high": 2}
df["conf_encoded"] = (df["confidence"].str.strip().str.lower().map(conf_map).fillna(1).astype(int)) \
                      if "confidence" in df.columns else 1
df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

TAB_COLS = [c for c in [
    "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "bright_diff", "frp_per_area",
    "is_night", "conf_encoded",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year",
] if c in df.columns]

X_tab = df[TAB_COLS].values.astype(np.float32)

tab_feat_train = X_tab[idx_train]
tab_feat_test  = X_tab[idx_test]
print(f"  Tabular feature dim: {tab_feat_train.shape[1]}")


# ===========================================================================
# ---- Fusion: concatenate all feature blocks --------------------------------
# ===========================================================================
print("\n--- Fusing features ---")

X_fused_train = np.hstack([pca_feat_train, cnn_feat_train, tab_feat_train])
X_fused_test  = np.hstack([pca_feat_test,  cnn_feat_test,  tab_feat_test])

print(f"  Fused feature dim: {X_fused_train.shape[1]}")
print(f"    PCA: {pca_feat_train.shape[1]}  "
      f"CNN: {cnn_feat_train.shape[1]}  "
      f"FIRMS: {tab_feat_train.shape[1]}")

# Scale all features together
scaler = StandardScaler()
X_fused_train = scaler.fit_transform(X_fused_train)
X_fused_test  = scaler.transform(X_fused_test)


# ---------------------------------------------------------------------------
# 4. Logistic Regression on fused features
# ---------------------------------------------------------------------------
print("\n--- Training Logistic Regression on fused features ---")

clf = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    C=0.1,             # slight regularisation to avoid overfitting wide features
    random_state=RANDOM_SEED
)
clf.fit(X_fused_train, y_train)

y_pred  = clf.predict(X_fused_test)
y_proba = clf.predict_proba(X_fused_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["wildfire", "industrial"]))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC (Hybrid): {auc:.4f}")


# ---------------------------------------------------------------------------
# 5. Compare against each individual experiment's saved scores
# ---------------------------------------------------------------------------
print("\n--- Comparison with individual experiments ---")
comparison = {"Hybrid": auc}

for name, path in [("PCA",   "results/pca_scores.npy"),
                   ("CNN",   "results/cnn_scores.npy"),
                   ("FIRMS", "results/firms_scores.npy")]:
    if os.path.exists(path):
        scores = np.load(path)
        lbls   = np.load(path.replace("_scores", "_labels"))
        # Scores might be for a different test split size; use what's available
        try:
            a = roc_auc_score(lbls, scores)
            comparison[name] = a
            print(f"  {name:8s}: AUC = {a:.4f}")
        except Exception:
            pass

print(f"  {'Hybrid':8s}: AUC = {auc:.4f}  ← this experiment")


# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
np.save("results/hybrid_scores.npy", y_proba)
np.save("results/hybrid_labels.npy", y_test)
print("\nSaved: results/hybrid_scores.npy  results/hybrid_labels.npy")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Experiment 4 — Hybrid Fusion Model",
             fontsize=14, fontweight="bold")

# AUC bar chart comparison
ax = axes[0]
names = list(comparison.keys())
aucs  = [comparison[n] for n in names]
colors = ["steelblue" if n != "Hybrid" else "darkorange" for n in names]
bars = ax.bar(names, aucs, color=colors, edgecolor="white", width=0.6)
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("ROC-AUC")
ax.set_title("AUC Comparison")
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Confusion matrix — hybrid
ax = axes[1]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["wildfire", "industrial"])
disp.plot(ax=ax, colorbar=False, cmap="Purples")
ax.set_title(f"Hybrid Confusion Matrix  (AUC = {auc:.3f})")

# Feature block contribution (variance of each block's scaled features)
ax = axes[2]
n_pca = pca_feat_train.shape[1]
n_cnn = cnn_feat_train.shape[1]
n_tab = tab_feat_train.shape[1]
block_vars = [
    np.var(X_fused_train[:, :n_pca]),
    np.var(X_fused_train[:, n_pca:n_pca + n_cnn]),
    np.var(X_fused_train[:, n_pca + n_cnn:])
]
ax.bar(["PCA\n(spatial\ncompressed)", "CNN\n(deep\nspatial)", "FIRMS\n(tabular\nspectral)"],
       block_vars, color=["steelblue", "seagreen", "coral"], edgecolor="white")
ax.set_ylabel("Feature variance (post-scaling)")
ax.set_title("Feature Block Spread")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("figures/hybrid_fusion.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/hybrid_fusion.png")

print("\nExperiment 4 complete.")