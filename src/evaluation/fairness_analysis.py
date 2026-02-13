"""
Issue #13: Fairness Analysis

Per-group metrics by PTGENDER and AGE bins.
Demographic parity, equalized odds, escalation rate disparity.
Saves: reports/tables/fairness_analysis.csv

Usage (from src/):
    python evaluation/fairness_analysis.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def _per_group_metrics(y_true, y_pred, escalated):
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) > 0 else float("nan")
    esc_rate = float(escalated.mean()) if len(escalated) > 0 else float("nan")
    return acc, f1, esc_rate


def run_fairness_analysis():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load escalation data
    esc_path = os.path.join(TABLES_DIR, "escalation_table_oof.csv")
    df_esc = pd.read_csv(esc_path)

    # Load preprocessed data and train indices for demographics
    data_path = os.path.join(PROJECT_ROOT, "data", "preprocessed_data.csv")
    preproc = pd.read_csv(data_path)
    train_indices = np.load(os.path.join(ARTIFACTS_DIR, "train_indices.npy"))

    # Align demographics with OOF escalation data
    demo = preproc.iloc[train_indices].reset_index(drop=True)
    n = min(len(demo), len(df_esc))
    demo = demo.iloc[:n]
    df_esc = df_esc.iloc[:n].copy()

    y_true = df_esc["true_label"].values.astype(int)
    y_pred = df_esc["pred_ens"].values.astype(int)
    escalated = (df_esc["escalation_level"] != "AI-Autonomous").values

    rows = []

    # Overall baseline
    acc, f1, esc_rate = _per_group_metrics(y_true, y_pred, escalated)
    rows.append({
        "group_variable": "overall",
        "group_value": "all",
        "n_samples": len(y_true),
        "accuracy": acc,
        "macro_f1": f1,
        "escalation_rate": esc_rate,
    })

    # By PTGENDER
    if "PTGENDER" in demo.columns:
        for g in sorted(demo["PTGENDER"].dropna().unique()):
            mask = (demo["PTGENDER"] == g).values[:n]
            if mask.sum() == 0:
                continue
            acc_g, f1_g, esc_g = _per_group_metrics(y_true[mask], y_pred[mask], escalated[mask])
            rows.append({
                "group_variable": "PTGENDER",
                "group_value": str(g),
                "n_samples": int(mask.sum()),
                "accuracy": acc_g,
                "macro_f1": f1_g,
                "escalation_rate": esc_g,
            })

    # By AGE bins
    if "AGE" in demo.columns:
        ages = demo["AGE"].values[:n]
        bins = [0, 65, 75, 85, 200]
        labels = ["<65", "65-74", "75-84", "85+"]
        age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
        for label in labels:
            mask = (age_groups == label).values if hasattr(age_groups, "values") else (age_groups == label)
            mask = np.asarray(mask)
            if mask.sum() == 0:
                continue
            acc_g, f1_g, esc_g = _per_group_metrics(y_true[mask], y_pred[mask], escalated[mask])
            rows.append({
                "group_variable": "AGE_bin",
                "group_value": label,
                "n_samples": int(mask.sum()),
                "accuracy": acc_g,
                "macro_f1": f1_g,
                "escalation_rate": esc_g,
            })

    df_out = pd.DataFrame(rows)

    # Compute disparity metrics
    if "PTGENDER" in demo.columns:
        gender_groups = df_out[df_out["group_variable"] == "PTGENDER"]
        if len(gender_groups) >= 2:
            esc_rates = gender_groups["escalation_rate"].values
            accs = gender_groups["accuracy"].values
            df_out.loc[df_out.index[-1], "demographic_parity_diff"] = float(max(esc_rates) - min(esc_rates))
            df_out.loc[df_out.index[-1], "equalized_odds_diff"] = float(max(accs) - min(accs))

    out_path = os.path.join(TABLES_DIR, "fairness_analysis.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved fairness analysis to: {out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_fairness_analysis()
