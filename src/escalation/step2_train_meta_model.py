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
from sklearn.metrics import roc_auc_score, classification_report
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
