"""
Components 3️⃣ 4️⃣ 5️⃣: Confusion matrix, ROC curves, calibration curves.
High-quality diagnostic plots for model audit (Step 1 of HITL pipeline).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


# ─────────────────────────────────────────────
# 3️⃣  Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrices (raw counts and normalized percentages side-by-side).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted integer-encoded labels.
    class_names : list of str
        Ordered class labels, e.g. ["SCD", "MCI", "AD"].
    save_path : str
        Output PNG file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_classes = len(class_names)

    # Raw and normalized matrices
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap_counts, cmap_norm = "Blues", "RdYlGn"

    # --- Counts ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap_counts,
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={'label': 'Count'}, square=True
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    # --- Normalized ---
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap=cmap_norm,
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={'label': 'Proportion'}, square=True
    )
    axes[1].set_title("Confusion Matrix (Normalized %)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"✅ Confusion matrix saved → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# 4️⃣  ROC Curves (One-vs-Rest)
# ─────────────────────────────────────────────
def plot_roc_ovr(y_true, y_prob, class_names, save_path):
    """
    Plot One-vs-Rest ROC curves for each class and macro-average ROC.

    Parameters
    ----------
    y_true : array-like (n_samples,)
    y_prob : array-like (n_samples, n_classes)
    class_names : list[str]
    save_path : str
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr, tpr, aucs = {}, {}, {}
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            aucs[i] = auc(fpr[i], tpr[i])
        except ValueError:
            aucs[i] = np.nan

    # Macro-average interpolation
    fpr_grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    valid = 0
    for i in range(n_classes):
        if i in fpr:
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
            valid += 1
    mean_tpr /= max(valid, 1)
    auc_macro = auc(fpr_grid, mean_tpr)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        if i in fpr:
            ax.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f"{class_names[i]} (AUC = {aucs[i]:.3f})"
            )

    # Macro-average + baseline
    ax.plot(fpr_grid, mean_tpr, 'k--', lw=2.5, label=f"Macro-Avg (AUC = {auc_macro:.3f})")
    ax.plot([0, 1], [0, 1], 'grey', ls='--', lw=1, alpha=0.5, label="Random Classifier")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"✅ ROC curves saved → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# 5️⃣  Calibration Curves
# ─────────────────────────────────────────────
def plot_calibration_curve(y_true, y_prob, class_names, save_path):
    """
    Plot reliability (calibration) curves per class.

    Parameters
    ----------
    y_true : array-like (n_samples,)
    y_prob : array-like (n_samples, n_classes)
    class_names : list[str]
    save_path : str
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # grid for subplots
    n_cols = min(3, n_classes)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        ax = axes[i]
        try:
            prob_true, prob_pred = calibration_curve(
                y_true_bin[:, i], y_prob[:, i], n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, "o-", color=colors[i], lw=2, label="Model")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
            ax.set_title(f"{class_names[i]} Calibration", fontsize=12, fontweight="bold")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{str(e)[:30]}", ha="center", va="center", fontsize=9)
            ax.set_axis_off()

    # turn off any unused subplots
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.text(0.5, 0.04, "Predicted Probability", ha="center", fontsize=11)
    fig.text(0.04, 0.5, "Observed Frequency", va="center", rotation="vertical", fontsize=11)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"✅ Calibration curves saved → {save_path}")
    plt.close(fig)
    return fig
