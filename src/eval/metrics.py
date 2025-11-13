"""
Component 1️⃣ 2️⃣: Overall and per-class performance metrics.
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


def compute_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute overall metrics: accuracy, precision, recall, F1, AUC, Brier score, log-loss.
    
    Parameters:
    ----------
    y_true : array-like
        True class labels (numeric 0..C-1).
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities from the model.
    class_names : list of str
        Ordered names of classes, e.g. ["SCD", "MCI", "AD"].

    Returns
    -------
    overall_metrics : dict
        Aggregate metrics (accuracy, macro_f1, weighted_f1, macro_auc, brier, logloss).
    per_class_df : pd.DataFrame
        DataFrame of per-class metrics (precision, recall, F1, AUC per class).
    """

    # --- Sanity check ---
    n_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # --- Binarize ground truth for multiclass AUC ---
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

    # --- Per-class metrics table ---
    per_class_df = per_class_metrics_table(y_true, y_pred, y_prob, class_names)

    return overall_metrics, per_class_df


def per_class_metrics_table(y_true, y_pred, y_prob, class_names):
    """
    Compute per-class precision, recall, F1, and AUC for each diagnostic stage.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted probabilities (n_samples, n_classes).
    class_names : list of str
        Ordered list of class names (e.g., ["SCD", "MCI", "AD"]).

    Returns
    -------
    pd.DataFrame
        Columns: ["Class", "Precision", "Recall", "F1", "AUC"]
    """

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
