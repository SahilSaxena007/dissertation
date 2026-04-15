# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Escalation quality vs tau / review budget.

This is the escalation-quality analogue of `evaluation/threshold_tradeoff_plot.py`.

We treat escalation as a binary classifier that tries to catch AI errors:
  Positive class = "AI is wrong"
  Score          = review_risk_score
  Decision       = escalate if score >= tau

Outputs:
  - reports/tables/escalation_quality_operating_point.csv
  - reports/tables/escalation_quality_sweep.csv
  - reports/tables/escalation_quality_budget_summary.csv
  - reports/figures/escalation_quality.png (if matplotlib is available)

Usage (from src/):
  .\\.venv\\Scripts\\python.exe evaluation/escalation_quality.py
  .\\.venv\\Scripts\\python.exe evaluation/escalation_quality.py --objective recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pandas'. Run with the project venv:\n"
        "  .\\src\\.venv\\Scripts\\python.exe evaluation/escalation_quality.py\n"
        "Or install deps:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'scikit-learn'. Run with the project venv:\n"
        "  .\\src\\.venv\\Scripts\\python.exe evaluation/escalation_quality.py"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_BUDGETS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]


def _read_current_tau() -> float:
    with open(ARTIFACTS_DIR / "step3_best_threshold.json") as f:
        best = json.load(f)
    return float(best["best_threshold"])


def metrics_at_tau(is_error: np.ndarray, risk: np.ndarray, tau: float) -> dict:
    escalated = risk >= tau
    tp = int(np.sum(escalated & is_error))
    fp = int(np.sum(escalated & ~is_error))
    fn = int(np.sum(~escalated & is_error))
    tn = int(np.sum(~escalated & ~is_error))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    review_rate = float(escalated.mean())

    return {
        "tau": float(tau),
        "review_rate": review_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "missed_error_rate": float(fn / max(tp + fn, 1)),  # 1 - recall
        "unnecessary_escalation_rate": float(fp / max(tp + fp, 1)),  # 1 - precision
    }


def _objective_key(objective: str) -> tuple[str, bool]:
    obj = objective.strip().lower()
    if obj in {"f1"}:
        return "f1", True
    if obj in {"recall", "tpr"}:
        return "recall", True
    if obj in {"precision"}:
        return "precision", True
    if obj in {"min_missed_errors", "missed_error_rate"}:
        return "missed_error_rate", False
    raise ValueError(f"Unknown objective: {objective}")


def budget_summary(sweep: pd.DataFrame, budgets: list[float], objective: str) -> pd.DataFrame:
    col, maximize = _objective_key(objective)
    rows: list[dict] = []
    for b in budgets:
        feasible = sweep[sweep["review_rate"] <= b]
        if feasible.empty:
            continue
        idx = feasible[col].idxmax() if maximize else feasible[col].idxmin()
        best = feasible.loc[idx]
        rows.append(
            {
                "budget": float(b),
                "best_threshold": round(float(best["tau"]), 3),
                "review_rate": round(float(best["review_rate"]), 3),
                "precision": round(float(best["precision"]), 4),
                "recall": round(float(best["recall"]), 4),
                "f1": round(float(best["f1"]), 4),
                "specificity": round(float(best["specificity"]), 4),
                "tp": int(best["tp"]),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
                "tn": int(best["tn"]),
                "objective": col,
                "objective_value": round(float(best[col]), 6),
            }
        )
    return pd.DataFrame(rows)


def plot(sweep: pd.DataFrame, is_error: np.ndarray, risk: np.ndarray, current_tau: float, budget_tbl: pd.DataFrame | None) -> Path | None:
    if plt is None:
        print("matplotlib is not available; skipped plot generation.")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # PR/ROC reflect the ranking quality of the risk score (tau-independent as a ranking).
    precision_curve, recall_curve, _ = precision_recall_curve(is_error.astype(int), risk)
    ap = float(average_precision_score(is_error.astype(int), risk))
    fpr, tpr, _ = roc_curve(is_error.astype(int), risk)
    roc_auc = float(roc_auc_score(is_error.astype(int), risk))

    op = sweep.loc[(sweep["tau"] - current_tau).abs().idxmin()].to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Escalation Quality: Risk Score as Error Detector", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(recall_curve, precision_curve, color="steelblue", linewidth=2, label=f"PR curve (AP={ap:.3f})")
    ax.axhline(float(is_error.mean()), color="gray", linestyle="--", linewidth=1.2, label=f"Random baseline ({is_error.mean():.2f})")
    ax.scatter([float(op["recall"])], [float(op["precision"])], color="darkorange", s=90, zorder=5, label=f"current tau~{float(op['tau']):.3f}")
    ax.set_xlabel("Recall (fraction of errors caught)", fontsize=11)
    ax.set_ylabel("Precision (fraction of escalations that are errors)", fontsize=11)
    ax.set_title("Precision-Recall", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    op_fpr = float(op["fp"]) / max(float(op["fp"]) + float(op["tn"]), 1.0)
    ax2.scatter([op_fpr], [float(op["recall"])], color="darkorange", s=90, zorder=5, label=f"current tau~{float(op['tau']):.3f}")
    ax2.set_xlabel("False Positive Rate (unnecessary escalations)", fontsize=11)
    ax2.set_ylabel("True Positive Rate (errors caught)", fontsize=11)
    ax2.set_title("ROC", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    sw = sweep.sort_values("review_rate")
    ax3.plot(sw["review_rate"] * 100, sw["precision"] * 100, color="#E8854C", linewidth=2, label="Precision")
    ax3.plot(sw["review_rate"] * 100, sw["recall"] * 100, color="#4C9BE8", linewidth=2, label="Recall")
    ax3.plot(sw["review_rate"] * 100, sw["f1"] * 100, color="#6DBE6D", linewidth=2, label="F1")
    ax3.axvline(float(op["review_rate"]) * 100, color="darkorange", linestyle="--", linewidth=1.5, label=f"Current (~{float(op['review_rate'])*100:.1f}%)")

    if budget_tbl is not None and not budget_tbl.empty:
        for _, row in budget_tbl.iterrows():
            if float(row["budget"]) in {0.08, 0.20, 0.40}:
                ax3.scatter([float(row["review_rate"]) * 100], [float(row["f1"]) * 100], color="purple", marker="D", s=55, zorder=5, alpha=0.75)

    ax3.set_xlabel("Review Rate (%)", fontsize=11)
    ax3.set_ylabel("Score (%)", fontsize=11)
    ax3.set_title("Scores vs Workload", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "escalation_quality.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="f1", help="Objective: f1 | recall | precision | min_missed_errors")
    parser.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS, help="Budget levels to summarize.")
    parser.add_argument("--sweep-points", type=int, default=201, help="Number of sweep points in [0,1] for tau grid (default: 201).")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TABLES_DIR / "threshold_data.csv")
    is_error = (~df["ai_correct"].astype(bool)).to_numpy(dtype=bool)
    risk = df["review_risk_score"].to_numpy(dtype=float)

    current_tau = _read_current_tau()

    sweep_rows: list[dict] = []
    for t in np.linspace(0.0, 1.0, int(args.sweep_points)):
        sweep_rows.append(metrics_at_tau(is_error, risk, float(t)))
    sweep = pd.DataFrame(sweep_rows)

    out_sweep = TABLES_DIR / "escalation_quality_sweep.csv"
    sweep.to_csv(out_sweep, index=False)
    print(f"Saved: {out_sweep}")

    op = sweep.loc[(sweep["tau"] - current_tau).abs().idxmin()].to_dict()
    op_out = TABLES_DIR / "escalation_quality_operating_point.csv"
    pd.DataFrame([op]).to_csv(op_out, index=False)
    print(f"Saved: {op_out}")

    budget_tbl = budget_summary(sweep, budgets=[float(b) for b in args.budgets], objective=str(args.objective))
    out_budget = TABLES_DIR / "escalation_quality_budget_summary.csv"
    budget_tbl.to_csv(out_budget, index=False)
    print("\n=== Best Threshold per Budget Level (Escalation Quality) ===")
    print(budget_tbl.to_string(index=False))
    print(f"Saved: {out_budget}")

    plot(sweep=sweep, is_error=is_error, risk=risk, current_tau=current_tau, budget_tbl=budget_tbl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

