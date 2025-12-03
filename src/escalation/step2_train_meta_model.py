"""
Step 2 – Train Reliability Meta-Model (Logistic Regression)

Reads:
    ../reports/tables/step2_meta_dataset.csv

Trains:
    LogisticRegression to predict ai_error (1 = AI wrong)

Saves:
    ../artifacts/step2_meta_model.pkl

Also creates:
    ../reports/tables/threshold_data.csv
with:
    sample_id, true_label, pred_ens, ai_correct,
    review_risk_score, and the risk features
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


def train_meta_model():
    tables_dir = os.path.join("..", "reports", "tables")
    artifacts_dir = os.path.join("..", "artifacts")

    meta_path = os.path.join(tables_dir, "step2_meta_dataset.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Could not find {meta_path}. "
            "Run `python .\\escalation\\step2_feature_builder.py` first."
        )

    print(f"📂 Loading meta-dataset from: {meta_path}")
    df = pd.read_csv(meta_path)

    feature_cols = [
        "risk_1_inv_conf",
        "risk_2_margin",
        "risk_3_entropy",
        "risk_4_disagreement",
        "risk_5_missing_fraction",
        "risk_6_critical_missing",
        "risk_7_multimodal_mismatch",
    ]

    X = df[feature_cols].values.astype(float)
    y = df["ai_error"].values.astype(int)

    print(f"🧮 Training LogisticRegression on {X.shape[0]} samples, {X.shape[1]} features …")
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="liblinear",
    )
    clf.fit(X, y)

    # basic evaluation on the same data (you can extend with CV if you want)
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob)
    print(f"\n📈 AUC (ai_error vs predicted risk): {auc:.3f}")
    print("\n📋 Classification report (threshold 0.5):")
    print(classification_report(y, y_pred, digits=3))

    # save model
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "step2_meta_model.pkl")
    joblib.dump(clf, model_path)
    print(f"\n💾 Saved meta-model to: {model_path}")

    # build threshold_data.csv for Step 3
    df_threshold = df[["sample_id", "true_label", "pred_ens", "ai_correct"]].copy()
    df_threshold["review_risk_score"] = y_prob

    # keep features too (optional but useful)
    for col in feature_cols:
        df_threshold[col] = df[col]

    threshold_path = os.path.join(tables_dir, "threshold_data.csv")
    df_threshold.to_csv(threshold_path, index=False)
    print(f"\n💾 Saved threshold dataset to: {threshold_path}")


if __name__ == "__main__":
    train_meta_model()
