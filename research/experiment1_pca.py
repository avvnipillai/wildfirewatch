# =============================================================================
# experiment1_pca.py
#
# PURPOSE: Use PCA (Principal Component Analysis) to classify thermal patches.
#
# HOW IT WORKS:
#   1. Flatten each 64×64 patch into a 4096-dim vector.
#   2. Fit PCA on the training set to find the main directions of variance.
#   3. Reconstruct every patch using only the top K components.
#   4. Compute reconstruction error (how badly PCA fails to reconstruct it).
#   5. Train a Logistic Regression on the PCA-compressed features.
#
# WHY PCA?
#   Industrial sources have a very regular, compact thermal pattern —
#   they project well onto a few principal components.
#   Wildfires are irregular and noisy — higher reconstruction error.
#   This difference in "compressibility" is a useful signal.
#
# OUTPUT:
#   results/pca_scores.npy  — predicted probabilities on the test set
#   results/pca_labels.npy  — true labels for the test set
#   figures/pca_components.png
#
# RUN: python experiment1_pca.py
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

RANDOM_SEED  = 42
N_COMPONENTS = 50    # how many PCA components to keep
TEST_SIZE    = 0.2

PATCHES_PATH = "data/patches.npy"
LABELS_PATH  = "data/patch_labels.npy"

os.makedirs("results",  exist_ok=True)
os.makedirs("figures",  exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading patches …")

if not os.path.exists(PATCHES_PATH):
    raise FileNotFoundError(
        f"{PATCHES_PATH} not found. Run step2_extract_patches.py first."
    )

patches = np.load(PATCHES_PATH)   # shape (N, 64, 64)
labels  = np.load(LABELS_PATH)    # shape (N,)

print(f"  Patches: {patches.shape}  Labels: {labels.shape}")
print(f"  Wildfire (0): {(labels==0).sum():,}   Industrial (1): {(labels==1).sum():,}")


# ---------------------------------------------------------------------------
# 2. Flatten patches: (N, 64, 64) → (N, 4096)
# ---------------------------------------------------------------------------
N = len(patches)
X = patches.reshape(N, -1).astype(np.float32)   # (N, 4096)
y = labels.astype(int)

print(f"\nFlattened shape: {X.shape}")


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}")


# ---------------------------------------------------------------------------
# 4. Fit PCA on training data
# ---------------------------------------------------------------------------
print(f"\nFitting PCA with {N_COMPONENTS} components on training set …")

pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

explained = pca.explained_variance_ratio_.cumsum()
print(f"  Variance explained by {N_COMPONENTS} components: "
      f"{100 * explained[-1]:.1f}%")


# ---------------------------------------------------------------------------
# 5. Reconstruction error as a feature
# ---------------------------------------------------------------------------
# The idea: reconstruct each sample from its PCA coordinates, then measure
# how different the reconstruction is from the original.
# Industrial sources reconstruct better (lower error).
# Wildfires are noisier so reconstruction error is higher.

def reconstruction_error(X_orig, X_pca, pca_model):
    """Mean squared error between original and PCA-reconstructed patches."""
    X_reconstructed = pca_model.inverse_transform(X_pca)
    err = np.mean((X_orig - X_reconstructed) ** 2, axis=1)  # shape (N,)
    return err

train_err = reconstruction_error(X_train, X_train_pca, pca)
test_err  = reconstruction_error(X_test,  X_test_pca,  pca)

print(f"\nReconstruction error (train):")
print(f"  Wildfire   mean: {train_err[y_train==0].mean():.6f}")
print(f"  Industrial mean: {train_err[y_train==1].mean():.6f}")


# ---------------------------------------------------------------------------
# 6. Build feature matrix: PCA coords + reconstruction error
# ---------------------------------------------------------------------------
# Combine compressed PCA features with the reconstruction error signal

X_train_feat = np.hstack([X_train_pca, train_err.reshape(-1, 1)])
X_test_feat  = np.hstack([X_test_pca,  test_err.reshape(-1, 1)])

# Scale features (Logistic Regression converges better with scaled inputs)
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat  = scaler.transform(X_test_feat)


# ---------------------------------------------------------------------------
# 7. Logistic Regression classifier
# ---------------------------------------------------------------------------
print("\nTraining Logistic Regression on PCA features …")

# class_weight='balanced' handles imbalanced datasets automatically
clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_SEED
)
clf.fit(X_train_feat, y_train)

y_pred  = clf.predict(X_test_feat)
y_proba = clf.predict_proba(X_test_feat)[:, 1]   # probability of class 1

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["wildfire", "industrial"]))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")


# ---------------------------------------------------------------------------
# 8. Save probability scores for the final comparison plot
# ---------------------------------------------------------------------------
np.save("results/pca_scores.npy", y_proba)
np.save("results/pca_labels.npy", y_test)
print("\nSaved: results/pca_scores.npy  results/pca_labels.npy")


# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Experiment 1 — PCA Analysis", fontsize=14, fontweight="bold")

# (a) Explained variance curve
ax = axes[0, 0]
ax.plot(np.arange(1, N_COMPONENTS + 1),
        pca.explained_variance_ratio_.cumsum() * 100, "b-o", ms=3)
ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative variance explained (%)")
ax.set_title("PCA Explained Variance")
ax.legend()
ax.grid(alpha=0.3)

# (b) First 4 principal components visualised as 64×64 images
comp_ax = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
for k, ax in enumerate(comp_ax):
    if k < N_COMPONENTS:
        comp_img = pca.components_[k].reshape(64, 64)
        im = ax.imshow(comp_img, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"PC {k+1}  ({100*pca.explained_variance_ratio_[k]:.1f}%)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

# (c) Reconstruction error distribution
ax = axes[1, 2]
ax.hist(train_err[y_train == 0], bins=40, alpha=0.6, label="Wildfire",   color="orange", density=True)
ax.hist(train_err[y_train == 1], bins=40, alpha=0.6, label="Industrial", color="steelblue", density=True)
ax.set_xlabel("Reconstruction error (MSE)")
ax.set_ylabel("Density")
ax.set_title("Reconstruction Error Distribution")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/pca_components.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/pca_components.png")

# Confusion matrix
fig2, ax2 = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["wildfire", "industrial"])
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"PCA Confusion Matrix  (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig("figures/pca_confusion.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/pca_confusion.png")

print("\nExperiment 1 complete.")