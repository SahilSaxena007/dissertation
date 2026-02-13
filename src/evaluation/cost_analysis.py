"""
Issue #11: Cost-Weighted Accuracy Analysis

Clinical cost matrix (missing AD is costliest).
Saves: reports/tables/cost_analysis.csv

Usage (from src/):
    python evaluation/cost_analysis.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")

# Clinical cost matrix: cost_matrix[true, predicted]
# SCD=0, MCI=1, AD=2
# Missing AD (predicting SCD/MCI when true is AD) is costliest
COST_MATRIX = np.array([
    # predicted: SCD  MCI  AD
    [0.0, 1.0, 2.0],   # true: SCD
    [1.5, 0.0, 1.0],   # true: MCI
    [5.0, 3.0, 0.0],   # true: AD  (missing AD -> SCD is worst)
])


def cost_weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: np.ndarray = COST_MATRIX,
) -> float:
    """
    Cost-weighted accuracy: 1 - (mean cost / max possible cost).
    Lower cost = higher accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    costs = np.array([cost_matrix[t, p] for t, p in zip(y_true, y_pred)])
    mean_cost = float(costs.mean())
    max_cost = float(cost_matrix.max())

    return 1.0 - (mean_cost / max_cost) if max_cost > 0 else 1.0


def total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: np.ndarray = COST_MATRIX,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(sum(cost_matrix[t, p] for t, p in zip(y_true, y_pred)))


def run_cost_analysis():
    os.makedirs(TABLES_DIR, exist_ok=True)

    thresh_path = os.path.join(TABLES_DIR, "threshold_data.csv")
    df = pd.read_csv(thresh_path)
    y_true = df["true_label"].values.astype(int)
    y_ai = df["pred_ens"].values.astype(int)
    risk_scores = df["review_risk_score"].values.astype(float)

    thresh_info = json.load(open(os.path.join(ARTIFACTS_DIR, "step3_best_threshold.json"), "r"))
    tau = float(thresh_info["best_threshold"])

    thresholds = np.linspace(0.0, 1.0, 51)
    h_accuracies = [0.85, 0.90, 0.95]
    rows = []

    for h_acc in h_accuracies:
        for tau_val in thresholds:
            escalated = risk_scores >= tau_val
            review_rate = float(escalated.mean())

            rng = np.random.default_rng(42)
            y_hitl = y_ai.copy()
            for i in np.where(escalated)[0]:
                if rng.random() < h_acc:
                    y_hitl[i] = y_true[i]
                else:
                    choices = [c for c in range(3) if c != y_true[i]]
                    y_hitl[i] = rng.choice(choices)

            rows.append({
                "human_accuracy": h_acc,
                "threshold": float(tau_val),
                "review_rate": review_rate,
                "ai_cost": total_cost(y_true, y_ai),
                "hitl_cost": total_cost(y_true, y_hitl),
                "ai_cost_weighted_acc": cost_weighted_accuracy(y_true, y_ai),
                "hitl_cost_weighted_acc": cost_weighted_accuracy(y_true, y_hitl),
                "cost_reduction": total_cost(y_true, y_ai) - total_cost(y_true, y_hitl),
            })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(TABLES_DIR, "cost_analysis.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved cost analysis to: {out_path}")

    # Summary at best threshold
    best = df_out[(df_out["threshold"] - tau).abs() < 0.02].groupby("human_accuracy").first()
    print("\nCost comparison at optimal threshold:")
    print(best[["ai_cost", "hitl_cost", "cost_reduction", "ai_cost_weighted_acc", "hitl_cost_weighted_acc"]].to_string())
    return df_out


if __name__ == "__main__":
    run_cost_analysis()
