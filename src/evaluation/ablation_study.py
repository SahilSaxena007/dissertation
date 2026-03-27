# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Issue #12: Ablation Study

Remove each risk feature one at a time, retrain meta-model, evaluate.
Saves: reports/tables/ablation_study.csv

Usage (from src/):
    python evaluation/ablation_study.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")

FEATURE_COLS = [
    "risk_1_inv_conf",
    "risk_2_margin",
    "risk_3_entropy",
    "risk_4_disagreement",
    "risk_5_missing_fraction",
    "risk_6_critical_missing",
    "risk_7_multimodal_mismatch",
]


def run_ablation_study():
    os.makedirs(TABLES_DIR, exist_ok=True)

    meta_path = os.path.join(TABLES_DIR, "step2_meta_dataset.csv")
    df = pd.read_csv(meta_path)

    X_full = df[FEATURE_COLS].values.astype(float)
    y = df["ai_error"].values.astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _evaluate(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=500, solver="liblinear")
        y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)
        return auc, acc

    # Full model baseline
    full_auc, full_acc = _evaluate(X_full, y)

    rows = [{
        "removed_feature": "NONE (full model)",
        "auc": full_auc,
        "accuracy": full_acc,
        "auc_change": 0.0,
        "accuracy_change": 0.0,
    }]

    # Ablation: remove one feature at a time
    for i, feat in enumerate(FEATURE_COLS):
        remaining = [j for j in range(len(FEATURE_COLS)) if j != i]
        X_ablated = X_full[:, remaining]
        abl_auc, abl_acc = _evaluate(X_ablated, y)

        rows.append({
            "removed_feature": feat,
            "auc": abl_auc,
            "accuracy": abl_acc,
            "auc_change": abl_auc - full_auc,
            "accuracy_change": abl_acc - full_acc,
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(TABLES_DIR, "ablation_study.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved ablation study to: {out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_ablation_study()
