# =============================================================================
# experiment3_firms.py
#
# PURPOSE: Classify wildfire vs industrial using ONLY the FIRMS tabular
#          attributes — no imagery needed.
#
# FEATURES USED:
#   - bright_ti4     : brightness temperature, channel I4 (~3.74 µm)
#   - bright_ti5     : brightness temperature, channel I5 (~11.45 µm)
#     FIRMS products using generic names are supported via brightness/bright_t31.
#   - frp            : Fire Radiative Power (MW) — energy output
#   - scan           : along-scan pixel size (related to satellite view angle)
#   - track           : along-track pixel size
#   - daynight       : D or N — fires behave differently at night
#   - confidence     : FIRMS detection confidence (nominal / low / high)
#   - hour_of_day    : extracted from acq_time (wildfires peak in afternoon)
#   - month          : extracted from acq_date (fire seasons)
#   - bright_diff    : bright_ti4 - bright_ti5 (fire vs ambient contrast)
#   - frp_per_area   : FRP / (scan × track) — power density
#
# MODELS:
#   - Random Forest (good default for tabular data)
#   - Gradient Boosting (XGBoost-style, sklearn implementation)
#   Both are compared; Random Forest score is saved for the final plot.
#
# OUTPUT:
#   results/firms_scores.npy   — probabilities on the test set
#   results/firms_labels.npy   — true labels for the test set
#   figures/firms_features.png — feature importance chart
#
# RUN: python experiment3_firms.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

RANDOM_SEED = 42
TEST_SIZE   = 0.2

LABELLED_CSV = "data/firms_labelled.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load labelled FIRMS data
# ---------------------------------------------------------------------------
print("Loading labelled FIRMS data …")

if not os.path.exists(LABELLED_CSV):
    raise FileNotFoundError(
        f"{LABELLED_CSV} not found. Run step1_label_dataset.py first."
    )

df = pd.read_csv(LABELLED_CSV)
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

print(f"  Loaded: {len(df):,} rows")
print(f"  Wildfire (0): {(df.label==0).sum():,}   "
      f"Industrial (1): {(df.label==1).sum():,}")


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("\nEngineering features …")

# FIRMS column names vary by product. The archive input can provide generic
# MODIS names instead of VIIRS-specific bright_ti4 / bright_ti5 names.
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

# Convert numeric columns safely
for col in ["bright_ti4", "bright_ti5", "frp", "scan", "track"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric values with the column median
num_cols = ["bright_ti4", "bright_ti5", "frp", "scan", "track"]
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Time features
if "acq_time" in df.columns:
    # acq_time is stored as HHMM integer (e.g. 1430 = 14:30)
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int)
    df["hour_of_day"] = df["acq_time"] // 100
else:
    df["hour_of_day"] = 12   # default to noon if not available

if "acq_date" in df.columns and df["acq_date"].notna().any():
    df["month"] = df["acq_date"].dt.month.fillna(6).astype(int)
    df["day_of_year"] = df["acq_date"].dt.dayofyear.fillna(180).astype(int)
else:
    df["month"] = 6
    df["day_of_year"] = 180

# Spectral difference: channel I4 - I5 separates hot sources from warm ground
df["bright_diff"] = df["bright_ti4"] - df["bright_ti5"]

# FRP per unit area: industrial sources have higher power density
pixel_area = df["scan"] * df["track"]   # approximate pixel area in km²
df["frp_per_area"] = df["frp"] / pixel_area.replace(0, np.nan)
df["frp_per_area"] = df["frp_per_area"].fillna(0)

# Encode day/night: D→0, N→1
if "daynight" in df.columns:
    df["is_night"] = (df["daynight"].str.strip().str.upper() == "N").astype(int)
else:
    df["is_night"] = 0

# Encode confidence: nominal→1, high→2, low→0
if "confidence" in df.columns:
    conf_map = {"low": 0, "nominal": 1, "high": 2}
    df["conf_encoded"] = (df["confidence"].str.strip().str.lower()
                            .map(conf_map)
                            .fillna(1)
                            .astype(int))
else:
    df["conf_encoded"] = 1

# Cyclic encoding of hour and month so the model sees continuity
# (hour 23 is close to hour 0; month 12 is close to month 1)
df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# All features to use
FEATURE_COLS = [
    "bright_ti4", "bright_ti5", "frp",
    "scan", "track",
    "bright_diff", "frp_per_area",
    "is_night", "conf_encoded",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_year",
]

# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["label"].values.astype(int)

print(f"\nFeature matrix shape: {X.shape}")


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")


# ---------------------------------------------------------------------------
# 4. Random Forest
# ---------------------------------------------------------------------------
print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",    # handles class imbalance
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)

print(classification_report(y_test, rf_pred,
                             target_names=["wildfire", "industrial"]))
print(f"ROC-AUC: {rf_auc:.4f}")

# 5-fold cross-validation on full dataset
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ---------------------------------------------------------------------------
# 5. Gradient Boosting (for comparison)
# ---------------------------------------------------------------------------
print("\n--- Gradient Boosting ---")

gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=RANDOM_SEED
)
gb.fit(X_train, y_train)

gb_pred  = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]
gb_auc   = roc_auc_score(y_test, gb_proba)

print(classification_report(y_test, gb_pred,
                             target_names=["wildfire", "industrial"]))
print(f"ROC-AUC: {gb_auc:.4f}")

print(f"\nSummary  |  Random Forest AUC: {rf_auc:.4f}  |  "
      f"Gradient Boosting AUC: {gb_auc:.4f}")


# ---------------------------------------------------------------------------
# 6. Save the best model's scores for the comparison plot
#    We save whichever model performed better on the test set.
# ---------------------------------------------------------------------------
if gb_auc >= rf_auc:
    best_proba = gb_proba
    best_name  = "Gradient Boosting"
else:
    best_proba = rf_proba
    best_name  = "Random Forest"

np.save("results/firms_scores.npy", best_proba)
np.save("results/firms_labels.npy", y_test)
print(f"\nSaved best FIRMS model ({best_name}) scores to results/firms_scores.npy")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Experiment 3 — FIRMS Feature Classification",
             fontsize=14, fontweight="bold")

# Feature importance from Random Forest
ax = axes[0]
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
ax.barh([FEATURE_COLS[i] for i in idx], importances[idx], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("RF Feature Importances")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Confusion matrix — Random Forest
ax = axes[1]
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["wildfire", "industrial"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Random Forest  (AUC = {rf_auc:.3f})")

# Confusion matrix — Gradient Boosting
ax = axes[2]
cm2 = confusion_matrix(y_test, gb_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                display_labels=["wildfire", "industrial"])
disp2.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Gradient Boosting  (AUC = {gb_auc:.3f})")

plt.tight_layout()
plt.savefig("figures/firms_features.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/firms_features.png")

# Distribution of key features by class
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
fig2.suptitle("Feature Distributions by Class", fontsize=13)

plot_features = ["bright_ti4", "bright_ti5", "frp", "bright_diff",
                 "frp_per_area", "hour_of_day"]
plot_features = [f for f in plot_features if f in df.columns]

for k, feat in enumerate(plot_features[:6]):
    ax = axes2.flat[k]
    wild_vals = df.loc[df.label == 0, feat].dropna()
    ind_vals  = df.loc[df.label == 1, feat].dropna()
    # Clip extreme outliers at the 99th percentile for readability
    clip_val = np.percentile(df[feat].dropna(), 99)
    ax.hist(wild_vals.clip(upper=clip_val), bins=40, alpha=0.6,
            label="Wildfire",   color="orange",    density=True)
    ax.hist(ind_vals.clip(upper=clip_val),  bins=40, alpha=0.6,
            label="Industrial", color="steelblue", density=True)
    ax.set_title(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/firms_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: figures/firms_distributions.png")

print("\nExperiment 3 complete.")