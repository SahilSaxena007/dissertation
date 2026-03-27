# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Components 1/2: Overall and per-class performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    Multiclass top-label Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if np.any(in_bin):
            bin_acc = correctness[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (np.sum(in_bin) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    Returns overall metrics dict (accuracy, F1, AUC, ECE, etc.)
    and per-class DataFrame.
    """
    n_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # binarize for multiclass AUC
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # --- Overall metrics ---
    overall_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }   

    # --- Macro AUC ---
    try:
        overall_metrics["macro_auc"] = roc_auc_score(
            y_true_bin, y_prob, average="macro", multi_class="ovr"
        )
    except ValueError:
        overall_metrics["macro_auc"] = np.nan

    # --- Calibration-oriented metrics ---
    try:
        overall_metrics["brier"] = brier_score_loss(y_true, np.max(y_prob, axis=1))
    except Exception:
        overall_metrics["brier"] = np.nan

    try:
        overall_metrics["logloss"] = log_loss(y_true, y_prob)
    except Exception:
        overall_metrics["logloss"] = np.nan

    try:
        overall_metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=15)
    except Exception:
        overall_metrics["ece"] = np.nan

    # --- Per-class metrics table ---
    per_class_df = per_class_metrics_table(y_true, y_pred, y_prob, class_names)

    return overall_metrics, per_class_df


def per_class_metrics_table(y_true, y_pred, y_prob, class_names):
    """Returns per-class precision, recall, F1, AUC as a DataFrame."""

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    aucs = []
    for i in range(n_classes):
        try:
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        except ValueError:
            auc_i = np.nan
        aucs.append(auc_i)

    per_class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": aucs
    })

    return per_class_df
