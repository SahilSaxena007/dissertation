import os
import numpy as np
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.calibration import calibration_curve, label_binarize
from sklearn.metrics import auc, brier_score_loss, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import seaborn as sns
from scipy.stats import entropy

def analyze_model_performance(y_true, y_pred, y_prob, model_name, class_names, output_dir="reports"):
    """
    Master function: orchestrates metric computation, CI estimation, visualization, and file saving.
    Returns a dictionary summary for programmatic use.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities
    - model_name: Name of the model (for saving files)
    - class_name: List of class names
    - output_dir: Directory to save reports and plots
    """
    metrics, per_class = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    f1_ci = bootstrap_confidence_intervals(
        metric_func=lambda yt, yp: f1_score(yt, yp, average='macro'),
        y_true=y_true,
        y_pred_or_proba=y_pred,
        n_bootstrap=1000,
        random_state=42
    )
    
    auc_ci = bootstrap_confidence_intervals(
        metric_func=lambda yt, yp: roc_auc_score(yt, yp, multi_class='ovr'),
        y_true=y_true,
        y_pred_or_proba=y_prob,
        n_bootstrap=1000,
        random_state=42
    )
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=f"{fig_dir}/confusion_matrix_{model_name}.png"
    )
    
    plot_roc_ovr(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        save_path=f"{fig_dir}/roc_ovr_{model_name}.png"
    )
    
    plot_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        save_path=f"{fig_dir}/calibration_{model_name}.png"
    )
    
    # 4️⃣ Export error taxonomy (for later HITL analysis)
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    
    export_error_taxonomy(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
        save_path=f"{table_dir}/error_taxonomy_{model_name}.csv"
    )
    
    # 5️⃣ Save detailed metrics tables
    save_metrics_tables(
        model_name=model_name,
        overall_dict=metrics,
        per_class_df=per_class,
        output_dir=table_dir
    )
    
    # 6️⃣ Construct a compact summary for console + dissertation appendix
    summary_dict = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_auc": metrics["macro_auc"],
        "brier": metrics.get("brier", None),
        "f1_ci": f1_ci,
        "auc_ci": auc_ci
    }

    print(f"\n✅ Finished analysis for {model_name}")
    print(f"Accuracy: {summary_dict['accuracy']:.3f} | Macro-F1: {summary_dict['macro_f1']:.3f} "
          f"(±{(f1_ci['ci_high']-f1_ci['ci_low'])/2:.3f}) | AUC: {summary_dict['macro_auc']:.3f}")

    return summary_dict
    
    
   
   
def compute_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute core classification metrics (overall + per-class) for a 3-class Alzheimer’s classifier.

    Parameters
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
        Aggregate metrics (accuracy, macro_f1, macro_auc, brier, logloss).
    per_class_df : pd.DataFrame
        DataFrame of per-class metrics: precision, recall, f1, auc.
    """

    # ----- Overall metrics -----
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")

    # Binarize true labels for multiclass AUC computation
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    try:
        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = np.nan  # if probs not proper shape or only one class present

    # Calibration-oriented metrics (for probabilistic quality)
    try:
        brier = brier_score_loss(y_true, np.max(y_prob, axis=1))
    except Exception:
        brier = np.nan
    try:
        logloss = log_loss(y_true, y_prob)
    except Exception:
        logloss = np.nan

    # ----- Per-class metrics -----
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # AUC per class (One-vs-Rest)
    aucs = []
    for i in range(len(class_names)):
        try:
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        except Exception:
            auc_i = np.nan
        aucs.append(auc_i)

    per_class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": aucs
    })

    overall_metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_auc": macro_auc,
        "brier": brier,
        "logloss": logloss
    }

    return overall_metrics, per_class_df
    
    
    
def bootstrap_confidence_intervals(
    metric_func,
    y_true,
    y_pred_or_proba,
    n_bootstrap=1000,
    random_state=42,
    ci=95
):
    """
    Estimate confidence intervals for any metric using non-parametric bootstrapping.

    Parameters
    ----------
    metric_func : callable
        Function that accepts (y_true, y_pred_or_proba) and returns a scalar metric value.
        e.g. lambda yt, yp: f1_score(yt, yp, average='macro')
    y_true : array-like
        True labels.
    y_pred_or_proba : array-like
        Predictions or probabilities (depends on metric_func).
    n_bootstrap : int
        Number of bootstrap samples.
    random_state : int
        Random seed for reproducibility.
    ci : float
        Confidence level (default 95%).

    Returns
    -------
    dict with keys: {"mean", "ci_low", "ci_high"}
    """

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    boot_metrics = []

    for i in range(n_bootstrap):
        # Sample indices with replacement
        idx = rng.randint(0, n, n)
        y_true_bs = np.array(y_true)[idx]
        y_pred_bs = np.array(y_pred_or_proba)[idx]

        try:
            metric_val = metric_func(y_true_bs, y_pred_bs)
            if np.isfinite(metric_val):
                boot_metrics.append(metric_val)
        except Exception:
            continue  # skip failed iterations (e.g. single-class sample)

    if len(boot_metrics) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    boot_metrics = np.array(boot_metrics)
    mean_val = np.mean(boot_metrics)
    lower = np.percentile(boot_metrics, (100 - ci) / 2)
    upper = np.percentile(boot_metrics, 100 - (100 - ci) / 2)

    return {"mean": mean_val, "ci_low": lower, "ci_high": upper}

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save a normalized confusion matrix (percentages).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Class labels in order, e.g. ["SCD", "MCI", "AD"].
    save_path : str
        File path to save the figure (e.g., './reports/figures/confusion_matrix_RF.png')
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # avoid NaNs when class has no samples

    plt.figure(figsize=(7, 6))
    sns.set(font_scale=1.3)
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        square=True
    )

    plt.title("Normalized Confusion Matrix", fontsize=18, pad=20)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)

    # Add counts inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5,
                    f"\n{cm[i, j]}",
                    ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_roc_ovr(y_true, y_prob, class_names, save_path):
    """
    Plot one-vs-rest ROC curves for a multi-class classifier.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (numeric 0..C-1).
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    class_names : list of str
        Class labels (e.g., ["SCD", "MCI", "AD"]).
    save_path : str
        File path to save figure.
    """
    # Binarize ground truth
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    # Macro-average curve
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, *roc_curve(y_true_bin[:, i], y_prob[:, i])[:2])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', lw=2.5,
             linestyle='--', label=f"Macro Avg (AUC = {macro_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=18)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    
def plot_calibration_curve(y_true, y_prob, class_names, save_path):
    """
    Plot reliability (calibration) curves for each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (numeric 0..C-1).
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    class_names : list of str
        Ordered class names.
    save_path : str
        Output file path for figure.
    """
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_prob[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', lw=2, label=f"{name}")

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Predicted Probability", fontsize=14)
    plt.ylabel("Observed Frequency", fontsize=14)
    plt.title("Calibration (Reliability) Curves", fontsize=18)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def export_error_taxonomy(y_true, y_pred, y_prob, class_names, model_name, save_path):
    """
    Export a detailed per-sample error taxonomy for post-hoc analysis.

    Parameters
    ----------
    y_true : array-like
        True class labels (numeric 0..C-1).
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities.
    class_names : list of str
        Class names in order of y_prob columns.
    model_name : str
        Model identifier for saving/logging.
    save_path : str
        Path to save CSV file.

    Output
    ------
    CSV table with columns:
        id | true_label | predicted_label | p_<class1> | p_<class2> | ... |
        max_prob | entropy | correct
    """
    n_classes = len(class_names)

    df = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred]
    })

    # Add probability columns
    for i, cname in enumerate(class_names):
        df[f"p_{cname}"] = y_prob[:, i]

    # Add uncertainty metrics
    df["max_prob"] = np.max(y_prob, axis=1)
    df["entropy"] = entropy(y_prob.T)  # per-sample entropy of predicted distribution
    df["correct"] = (y_true == y_pred).astype(int)

    # Add a unique ID if you like to link back to patient index
    df["id"] = np.arange(len(df))

    df.to_csv(save_path, index=False)
    print(f"🧾 Saved error taxonomy for {model_name} → {save_path}")
    
    
def save_metrics_tables(model_name, overall_dict, per_class_df, output_dir):
    """
    Save model performance metrics to CSV (overall + per-class).

    Parameters
    ----------
    model_name : str
        Name of model (e.g., 'CatBoost').
    overall_dict : dict
        Overall metrics from compute_classification_metrics.
    per_class_df : pd.DataFrame
        DataFrame of per-class metrics.
    output_dir : str
        Folder path to save CSVs.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Overall metrics → one-row DataFrame
    overall_df = pd.DataFrame([overall_dict])
    overall_df.insert(0, "Model", model_name)
    overall_path = os.path.join(output_dir, f"performance_overall_{model_name}.csv")
    overall_df.to_csv(overall_path, index=False)

    # Per-class metrics
    per_class = per_class_df.copy()
    per_class.insert(0, "Model", model_name)
    perclass_path = os.path.join(output_dir, f"performance_perclass_{model_name}.csv")
    per_class.to_csv(perclass_path, index=False)

    print(f"💾 Saved metrics tables for {model_name}:")
    print(f"   Overall → {overall_path}")
    print(f"   Per-class → {perclass_path}")