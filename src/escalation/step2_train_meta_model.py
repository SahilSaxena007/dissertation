"""
Step 2 - Train Reliability Meta-Model (Model Comparison)

Reads:
    ../reports/tables/step2_meta_dataset.csv

Trains & Compares:
    LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
    Selects best by 5-fold CV AUC on ai_error prediction.

Evaluation:
    Stratified K-Fold out-of-sample predictions (no train/eval overlap)

Saves:
    ../artifacts/step2_meta_model.pkl          (best model)
    ../reports/tables/threshold_data.csv
    ../reports/tables/step2_meta_model_comparison.csv
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


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


def _build_candidate_models() -> dict:
    """Return named candidate meta-models for comparison."""
    return {
        "LogisticRegression": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }


def _compare_meta_models(X: np.ndarray, y: np.ndarray, cv) -> pd.DataFrame:
    """
    Compare candidate meta-models via stratified CV.

    Returns a DataFrame with per-model CV AUC, accuracy, and classification
    reports, sorted by AUC descending.
    """
    candidates = _build_candidate_models()
    rows = []

    for name, clf in candidates.items():
        print(f"\n  Evaluating meta-model candidate: {name}")
        y_prob_oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred_oof = (y_prob_oof >= 0.5).astype(int)

        auc = roc_auc_score(y, y_prob_oof)
        acc = accuracy_score(y, y_pred_oof)

        print(f"    CV AUC  = {auc:.4f}")
        print(f"    CV Acc  = {acc:.4f}")

        rows.append({
            "model": name,
            "cv_auc": float(auc),
            "cv_accuracy": float(acc),
        })

    df_cmp = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    return df_cmp


def train_meta_model(meta_path: str | None = None):
    tables_dir = TABLES_DIR
    artifacts_dir = ARTIFACTS_DIR

    if meta_path is None:
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Model comparison ─────────────────────────────────────────────
    print(f"\nComparing 3 candidate meta-models on {X.shape[0]} samples (5-fold CV)...")
    df_cmp = _compare_meta_models(X, y, cv)

    cmp_path = os.path.join(tables_dir, "step2_meta_model_comparison.csv")
    df_cmp.to_csv(cmp_path, index=False)
    print(f"\nModel comparison saved to: {cmp_path}")
    print(df_cmp.to_string(index=False))

    best_model_name = df_cmp.iloc[0]["model"]
    best_cv_auc = df_cmp.iloc[0]["cv_auc"]
    print(f"\nBest meta-model: {best_model_name} (CV AUC = {best_cv_auc:.4f})")

    # ── Generate OOF risk scores with best model ─────────────────────
    candidates = _build_candidate_models()
    best_clf = candidates[best_model_name]

    print(f"\nGenerating OOF risk scores with {best_model_name}...")
    y_prob_oof = cross_val_predict(best_clf, X, y, cv=cv, method="predict_proba")[:, 1]
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

    # ── Fit final model on full data for deployment ──────────────────
    final_clf = candidates[best_model_name]  # fresh instance (same hyperparams)
    final_clf.fit(X, y)

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "step2_meta_model.pkl")
    joblib.dump(final_clf, model_path)
    print(f"Saved meta-model ({best_model_name}) to: {model_path}")

    # Save selection metadata alongside model
    selection_info = {
        "selected_model": best_model_name,
        "cv_auc": float(best_cv_auc),
        "comparison": df_cmp.to_dict(orient="records"),
    }
    selection_path = os.path.join(artifacts_dir, "step2_meta_model_selection.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_info, f, indent=2)
    print(f"Saved selection metadata to: {selection_path}")

    # threshold_data now uses CV out-of-sample risk scores (leak-free)
    df_threshold = df[["sample_id", "true_label", "pred_ens", "ai_correct"]].copy()
    df_threshold["review_risk_score"] = y_prob_oof
    for col in FEATURE_COLS:
        df_threshold[col] = df[col]

    threshold_path = os.path.join(tables_dir, "threshold_data.csv")
    df_threshold.to_csv(threshold_path, index=False)
    print(f"Saved threshold dataset to: {threshold_path}")
    return {
        "model_path": model_path,
        "threshold_data_path": threshold_path,
        "bootstrap_metrics_path": boot_path,
        "comparison_path": cmp_path,
        "selected_model": best_model_name,
        "cv_auc": float(auc),
        "n_samples": int(X.shape[0]),
    }


if __name__ == "__main__":
    train_meta_model()
