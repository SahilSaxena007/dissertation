"""
Step 2 - Train Reliability Meta-Model (Logistic Regression)

Reads:
    ../reports/tables/step2_meta_dataset.csv

Trains:
    LogisticRegression to predict ai_error (1 = AI wrong)

Evaluation:
    Stratified K-Fold out-of-sample predictions (no train/eval overlap)

Saves:
    ../artifacts/step2_meta_model.pkl
    ../reports/tables/threshold_data.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


FEATURE_COLS = [
    "risk_1_inv_conf",
    "risk_2_margin",
    "risk_3_entropy",
    "risk_4_disagreement",
    "risk_5_missing_fraction",
    "risk_6_critical_missing",
    "risk_7_multimodal_mismatch",
]


def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample indices with replacement while preserving class counts.
    """
    y = np.asarray(y)
    sampled = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        sampled_cls = rng.choice(cls_idx, size=len(cls_idx), replace=True)
        sampled.append(sampled_cls)

    idx = np.concatenate(sampled)
    rng.shuffle(idx)
    return idx


def compute_bootstrap_meta_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified bootstrap CIs for meta-model OOF evaluation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    rng = np.random.default_rng(seed)
    auc_vals = []
    acc_vals = []

    for _ in range(n_bootstraps):
        b_idx = _stratified_bootstrap_indices(y_true, rng)
        y_b = y_true[b_idx]
        p_b = y_prob[b_idx]
        pred_b = y_pred[b_idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_vals.append(roc_auc_score(y_b, p_b))
        acc_vals.append(accuracy_score(y_b, pred_b))

    def summarize(metric_name: str, vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=float)
        return {
            "metric": metric_name,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "ci_lower_95": float(np.percentile(arr, 2.5)),
            "ci_upper_95": float(np.percentile(arr, 97.5)),
            "n_bootstraps_used": int(arr.size),
        }

    return pd.DataFrame(
        [
            summarize("auc", auc_vals),
            summarize("accuracy_at_0.5", acc_vals),
        ]
    )


def train_meta_model():
    tables_dir = os.path.join("..", "reports", "tables")
    artifacts_dir = os.path.join("..", "artifacts")

    meta_path = os.path.join(tables_dir, "step2_meta_dataset.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Could not find {meta_path}. "
            "Run `python .\\escalation\\step2_feature_builder.py` first."
        )

    print(f"Loading meta-dataset from: {meta_path}")
    df = pd.read_csv(meta_path)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["ai_error"].values.astype(int)

    base_clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="liblinear",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Running 5-fold CV for out-of-sample meta risk scores on {X.shape[0]} samples...")
    y_prob_oof = cross_val_predict(base_clf, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_oof = (y_prob_oof >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob_oof)
    print(f"CV AUC (ai_error vs predicted risk): {auc:.3f}")
    print("Classification report on CV out-of-sample predictions:")
    print(classification_report(y, y_pred_oof, digits=3))

    # Stratified bootstrap over OOF predictions for robust small-sample estimates
    df_boot = compute_bootstrap_meta_metrics(y_true=y, y_prob=y_prob_oof, n_bootstraps=1000, seed=42)
    boot_path = os.path.join(tables_dir, "step2_meta_bootstrap_metrics.csv")
    df_boot.to_csv(boot_path, index=False)
    print(f"Saved stratified bootstrap metrics to: {boot_path}")

    # Fit final model on full data for deployment/inference use
    final_clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="liblinear",
    )
    final_clf.fit(X, y)

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "step2_meta_model.pkl")
    joblib.dump(final_clf, model_path)
    print(f"Saved meta-model to: {model_path}")

    # threshold_data now uses CV out-of-sample risk scores (leak-free)
    df_threshold = df[["sample_id", "true_label", "pred_ens", "ai_correct"]].copy()
    df_threshold["review_risk_score"] = y_prob_oof
    for col in FEATURE_COLS:
        df_threshold[col] = df[col]

    threshold_path = os.path.join(tables_dir, "threshold_data.csv")
    df_threshold.to_csv(threshold_path, index=False)
    print(f"Saved threshold dataset to: {threshold_path}")


if __name__ == "__main__":
    train_meta_model()
