# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Step 3 - Threshold Optimisation for HITL Policy.

Reads:
    ../reports/tables/threshold_data.csv

Computes:
    - Threshold analysis for a fixed human accuracy (legacy-compatible mode)
    - Sensitivity analysis across a human-accuracy distribution

Saves:
    ../reports/tables/threshold_analysis.csv
    ../reports/tables/threshold_analysis_by_human_accuracy.csv
    ../reports/tables/threshold_sensitivity_summary.csv
    ../artifacts/step3_best_threshold.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def _load_threshold_inputs() -> Tuple[np.ndarray, np.ndarray]:
    thresh_path = os.path.join(TABLES_DIR, "threshold_data.csv")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(
            f"Could not find {thresh_path}. "
            "Run `python .\\escalation\\step2_train_meta_model.py` first."
        )

    print(f"Loading threshold dataset from: {thresh_path}")
    df = pd.read_csv(thresh_path)

    if "review_risk_score" not in df.columns or "ai_correct" not in df.columns:
        raise ValueError("threshold_data.csv must contain 'review_risk_score' and 'ai_correct' columns.")

    risk = df["review_risk_score"].values.astype(float)
    ai_correct = df["ai_correct"].values.astype(int)
    return risk, ai_correct


def _build_threshold_curve(
    risk: np.ndarray,
    ai_correct: np.ndarray,
    human_accuracy: float,
    num_thresholds: int,
) -> pd.DataFrame:
    if not (0.0 < human_accuracy <= 1.0):
        raise ValueError("human_accuracy must be in the interval (0, 1].")

    ai_only_acc = float(ai_correct.mean())
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    rows = []

    for tau in thresholds:
        escalate = risk >= tau
        review_rate = float(escalate.mean())

        final_correct = np.where(escalate, human_accuracy, ai_correct.astype(float))
        policy_acc = float(final_correct.mean())
        accuracy_gain = float(policy_acc - ai_only_acc)

        rows.append(
            {
                "threshold": float(tau),
                "review_rate": review_rate,
                "policy_accuracy": policy_acc,
                "ai_only_accuracy": ai_only_acc,
                "accuracy_gain": accuracy_gain,
                "human_accuracy_assumed": float(human_accuracy),
            }
        )

    return pd.DataFrame(rows)


def _pick_best_under_budget(df_tau: pd.DataFrame, review_budget: float) -> pd.Series:
    feasible = df_tau[df_tau["review_rate"] <= review_budget]
    if feasible.empty:
        print("No threshold satisfies review budget; using global best policy.")
        return df_tau.loc[df_tau["policy_accuracy"].idxmax()]
    return feasible.loc[feasible["policy_accuracy"].idxmax()]


def optimize_thresholds(
    review_budget: float = 0.4,
    human_accuracy: float = 0.95,
    num_thresholds: int = 201,
) -> Dict[str, Any]:
    """
    Legacy-compatible single-assumption optimization.
    """
    risk, ai_correct = _load_threshold_inputs()
    df_tau = _build_threshold_curve(
        risk=risk,
        ai_correct=ai_correct,
        human_accuracy=human_accuracy,
        num_thresholds=num_thresholds,
    )

    tables_dir = TABLES_DIR
    artifacts_dir = ARTIFACTS_DIR
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    out_analysis = os.path.join(tables_dir, "threshold_analysis.csv")
    df_tau.to_csv(out_analysis, index=False)
    print(f"Saved threshold analysis to: {out_analysis}")

    best_row = _pick_best_under_budget(df_tau, review_budget=review_budget)
    best_info = {
        "best_threshold": float(best_row["threshold"]),
        "review_budget": float(review_budget),
        "review_rate": float(best_row["review_rate"]),
        "policy_accuracy": float(best_row["policy_accuracy"]),
        "ai_only_accuracy": float(best_row["ai_only_accuracy"]),
        "accuracy_gain": float(best_row["accuracy_gain"]),
        "human_accuracy_assumed": float(best_row["human_accuracy_assumed"]),
        "selection_mode": "fixed_human_accuracy",
    }

    best_path = os.path.join(artifacts_dir, "step3_best_threshold.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"Saved best threshold info to: {best_path}")

    return best_info


def optimize_thresholds_with_sensitivity(
    review_budget: float = 0.4,
    human_accuracy_values: Iterable[float] | None = None,
    num_thresholds: int = 201,
) -> Dict[str, Any]:
    """
    Sensitivity-aware optimization using a human-accuracy distribution.

    Default distribution:
      Uniform grid in [0.85, 0.95] with 0.01 step.
    """
    if human_accuracy_values is None:
        human_accuracy_values = np.round(np.arange(0.85, 0.951, 0.01), 2).tolist()

    h_values = np.array(sorted(set(float(v) for v in human_accuracy_values)), dtype=float)
    if h_values.size == 0:
        raise ValueError("human_accuracy_values cannot be empty.")
    if np.any(h_values <= 0.0) or np.any(h_values > 1.0):
        raise ValueError("All human_accuracy_values must be in the interval (0, 1].")

    risk, ai_correct = _load_threshold_inputs()
    ai_only_acc = float(ai_correct.mean())
    print(f"Baseline AI-only accuracy: {ai_only_acc:.3f}")
    print(
        "Running sensitivity sweep over human accuracy values: "
        + ", ".join(f"{v:.2f}" for v in h_values)
    )

    df_all = []
    per_scenario_best = []
    for h_acc in h_values:
        df_tau = _build_threshold_curve(
            risk=risk,
            ai_correct=ai_correct,
            human_accuracy=float(h_acc),
            num_thresholds=num_thresholds,
        )
        df_all.append(df_tau)
        best_row = _pick_best_under_budget(df_tau, review_budget=review_budget)
        per_scenario_best.append(
            {
                "human_accuracy_assumed": float(h_acc),
                "best_threshold": float(best_row["threshold"]),
                "review_rate": float(best_row["review_rate"]),
                "policy_accuracy": float(best_row["policy_accuracy"]),
                "ai_only_accuracy": float(best_row["ai_only_accuracy"]),
                "accuracy_gain": float(best_row["accuracy_gain"]),
            }
        )

    df_all_tau = pd.concat(df_all, ignore_index=True)

    # Distribution-level expectation for each threshold (uniform over provided values)
    grouped = (
        df_all_tau.groupby("threshold", as_index=False)
        .agg(
            review_rate=("review_rate", "mean"),
            expected_policy_accuracy=("policy_accuracy", "mean"),
            min_policy_accuracy=("policy_accuracy", "min"),
            max_policy_accuracy=("policy_accuracy", "max"),
            ai_only_accuracy=("ai_only_accuracy", "mean"),
        )
    )
    grouped["expected_accuracy_gain"] = grouped["expected_policy_accuracy"] - grouped["ai_only_accuracy"]
    grouped["human_accuracy_distribution"] = f"Uniform[{h_values.min():.2f},{h_values.max():.2f}]"

    feasible = grouped[grouped["review_rate"] <= review_budget]
    if feasible.empty:
        print("No threshold satisfies review budget; using global best expected policy.")
        robust_best = grouped.loc[grouped["expected_policy_accuracy"].idxmax()]
    else:
        robust_best = feasible.loc[feasible["expected_policy_accuracy"].idxmax()]

    tables_dir = TABLES_DIR
    artifacts_dir = ARTIFACTS_DIR
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    out_expected = os.path.join(tables_dir, "threshold_analysis.csv")
    out_all = os.path.join(tables_dir, "threshold_analysis_by_human_accuracy.csv")
    out_summary = os.path.join(tables_dir, "threshold_sensitivity_summary.csv")

    grouped.to_csv(out_expected, index=False)
    df_all_tau.to_csv(out_all, index=False)
    pd.DataFrame(per_scenario_best).to_csv(out_summary, index=False)

    print(f"Saved expected threshold analysis to: {out_expected}")
    print(f"Saved full sensitivity curves to: {out_all}")
    print(f"Saved per-scenario best-threshold summary to: {out_summary}")

    best_info = {
        "best_threshold": float(robust_best["threshold"]),
        "review_budget": float(review_budget),
        "review_rate": float(robust_best["review_rate"]),
        "policy_accuracy": float(robust_best["expected_policy_accuracy"]),
        "ai_only_accuracy": float(robust_best["ai_only_accuracy"]),
        "accuracy_gain": float(robust_best["expected_accuracy_gain"]),
        "human_accuracy_assumed": float(h_values.mean()),
        "human_accuracy_range": [float(h_values.min()), float(h_values.max())],
        "human_accuracy_distribution": f"Uniform[{h_values.min():.2f},{h_values.max():.2f}]",
        "selection_mode": "expected_policy_accuracy_under_sensitivity",
        "worst_case_policy_accuracy": float(robust_best["min_policy_accuracy"]),
        "best_case_policy_accuracy": float(robust_best["max_policy_accuracy"]),
    }

    best_path = os.path.join(artifacts_dir, "step3_best_threshold.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"Saved robust best threshold info to: {best_path}")

    return best_info


if __name__ == "__main__":
    optimize_thresholds_with_sensitivity()
