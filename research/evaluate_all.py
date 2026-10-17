# =============================================================================
# evaluate_all.py
#
# PURPOSE: Load the saved probability scores from all 4 experiments and
#          produce a single comparison figure:
#           - ROC curves for all 4 methods on one plot
#           - Precision-Recall curves
#           - AUC bar chart
#           - Per-model score distribution
#
# All 4 experiments save:
#   results/<name>_scores.npy   — predicted probabilities (positive class)
#   results/<name>_labels.npy   — true binary labels
#
# RUN: python evaluate_all.py
#      (Run AFTER all 4 experiment scripts have been executed.)
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report
)

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Experiments to compare
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "PCA"    : ("results/pca_scores.npy",    "results/pca_labels.npy",    "steelblue",  "o"),
    "CNN"    : ("results/cnn_scores.npy",    "results/cnn_labels.npy",    "seagreen",   "s"),
    "FIRMS"  : ("results/firms_scores.npy",  "results/firms_labels.npy",  "darkorange", "^"),
    "Hybrid" : ("results/hybrid_scores.npy", "results/hybrid_labels.npy", "crimson",    "D"),
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results = {}
missing = []

for name, (score_path, label_path, color, marker) in EXPERIMENTS.items():
    if os.path.exists(score_path) and os.path.exists(label_path):
        scores = np.load(score_path)
        labels = np.load(label_path)
        results[name] = {
            "scores" : scores,
            "labels" : labels.astype(int),
            "color"  : color,
            "marker" : marker,
        }
        print(f"  Loaded {name}: {len(scores):,} samples  "
              f"(wildfire: {(labels==0).sum():,}  "
              f"industrial: {(labels==1).sum():,})")
    else:
        missing.append(name)

if missing:
    print(f"\nWARNING: Results not found for: {missing}")
    print("Run the missing experiment scripts before running evaluate_all.py.")

if not results:
    raise RuntimeError(
        "No experiment results found. Run at least one experiment first."
    )


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"{'Model':<10} {'ROC-AUC':>8} {'Avg Precision':>14} {'F1 (thresh=0.5)':>16}")
print("-" * 52)

for name, d in results.items():
    y_true  = d["labels"]
    y_score = d["scores"]

    roc_auc = roc_auc_score(y_true, y_score)
    avg_pre = average_precision_score(y_true, y_score)

    y_pred = (y_score >= 0.5).astype(int)
    # F1 for the positive (industrial) class
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

    d["roc_auc"] = roc_auc
    d["avg_pre"] = avg_pre
    d["f1"]      = f1

    print(f"  {name:<10} {roc_auc:>8.4f} {avg_pre:>14.4f} {f1:>16.4f}")


# ---------------------------------------------------------------------------
# Figure: 4-panel comparison
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ---- Panel 1: ROC curves ----
ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.set_title("ROC Curves — All Experiments", fontsize=12, fontweight="bold")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.50)")

for name, d in results.items():
    fpr, tpr, _ = roc_curve(d["labels"], d["scores"])
    ax_roc.plot(fpr, tpr,
                color=d["color"],
                linewidth=2,
                label=f"{name}  (AUC = {d['roc_auc']:.3f})")

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc="lower right", fontsize=9)
ax_roc.grid(alpha=0.3)
ax_roc.set_xlim(-0.01, 1.01)
ax_roc.set_ylim(-0.01, 1.01)


# ---- Panel 2: Precision-Recall curves ----
ax_pr = fig.add_subplot(gs[0, 1])
ax_pr.set_title("Precision-Recall Curves — All Experiments",
                fontsize=12, fontweight="bold")

# Baseline: always predict positive → precision = class prevalence
first_labels = next(iter(results.values()))["labels"]
baseline_pre = first_labels.mean()
ax_pr.axhline(baseline_pre, color="black", linestyle="--", linewidth=1,
              alpha=0.5, label=f"Random (AP = {baseline_pre:.2f})")

for name, d in results.items():
    pre, rec, _ = precision_recall_curve(d["labels"], d["scores"])
    ax_pr.plot(rec, pre,
               color=d["color"],
               linewidth=2,
               label=f"{name}  (AP = {d['avg_pre']:.3f})")

ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(loc="upper right", fontsize=9)
ax_pr.grid(alpha=0.3)
ax_pr.set_xlim(-0.01, 1.01)
ax_pr.set_ylim(-0.01, 1.05)


# ---- Panel 3: AUC bar chart ----
ax_bar = fig.add_subplot(gs[1, 0])
ax_bar.set_title("ROC-AUC Comparison", fontsize=12, fontweight="bold")

names  = list(results.keys())
aucs   = [results[n]["roc_auc"] for n in names]
colors = [results[n]["color"]   for n in names]

bars = ax_bar.bar(names, aucs, color=colors, edgecolor="white", width=0.55)
ax_bar.set_ylim(max(0.4, min(aucs) - 0.1), min(1.02, max(aucs) + 0.08))
ax_bar.set_ylabel("ROC-AUC")
ax_bar.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.4,
               label="Random baseline")
ax_bar.legend(fontsize=9)
ax_bar.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, aucs):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")


# ---- Panel 4: Score distributions ----
ax_dist = fig.add_subplot(gs[1, 1])
ax_dist.set_title("Predicted Score Distributions", fontsize=12, fontweight="bold")

# For readability, show only one model's score split by true class.
# Pick the best model (highest AUC).
best_name = max(results, key=lambda n: results[n]["roc_auc"])
d_best    = results[best_name]

wild_scores = d_best["scores"][d_best["labels"] == 0]
ind_scores  = d_best["scores"][d_best["labels"] == 1]

ax_dist.hist(wild_scores, bins=40, alpha=0.6, label="Wildfire (true 0)",
             color="darkorange",  density=True)
ax_dist.hist(ind_scores,  bins=40, alpha=0.6, label="Industrial (true 1)",
             color="steelblue",   density=True)
ax_dist.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                label="Threshold = 0.5")
ax_dist.set_xlabel(f"Predicted probability ({best_name} model)")
ax_dist.set_ylabel("Density")
ax_dist.legend(fontsize=9)
ax_dist.grid(alpha=0.3)

# Overall figure title
fig.suptitle("Wildfire vs Industrial Heat — Experiment Comparison\n"
             f"({', '.join(results.keys())})",
             fontsize=14, fontweight="bold", y=1.01)

plt.savefig("figures/all_experiments_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: figures/all_experiments_comparison.png")


# ---------------------------------------------------------------------------
# Print full classification report for the best model
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["roc_auc"])
d_best    = results[best_name]
print(f"\n=== Full report for best model: {best_name} ===")
print(classification_report(
    d_best["labels"],
    (d_best["scores"] >= 0.5).astype(int),
    target_names=["wildfire", "industrial"]
))


# ---------------------------------------------------------------------------
# Print a summary table
# ---------------------------------------------------------------------------
print("=== Final Summary Table ===")
print(f"{'Model':<12} {'ROC-AUC':>9} {'Avg Prec':>10} {'F1 (ind)':>10}")
print("=" * 45)
for name in sorted(results, key=lambda n: results[n]["roc_auc"], reverse=True):
    d = results[name]
    star = " ← best" if name == best_name else ""
    print(f"  {name:<12} {d['roc_auc']:>9.4f} {d['avg_pre']:>10.4f} "
          f"{d['f1']:>10.4f}{star}")

print("\nDone. Open figures/all_experiments_comparison.png for the full comparison.")