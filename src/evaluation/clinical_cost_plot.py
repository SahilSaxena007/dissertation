# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Clinical cost trade-off vs tau / review budget.

This is the clinical-cost analogue of `evaluation/threshold_tradeoff_plot.py`.
It reads the full sweep from `reports/tables/cost_analysis.csv` and:
  1) prints a "best tau per budget" table under a chosen objective
  2) saves a trade-off plot to `reports/figures/clinical_cost_tradeoff.png`
  3) saves operating-point summaries to `reports/tables/clinical_cost_summary.csv`

Inputs:
  - reports/tables/cost_analysis.csv
  - artifacts/step3_best_threshold.json

Usage (from src/):
  .\\.venv\\Scripts\\python.exe evaluation/clinical_cost_plot.py
  .\\.venv\\Scripts\\python.exe evaluation/clinical_cost_plot.py --objective cost_reduction --human-acc 0.90
  .\\.venv\\Scripts\\python.exe evaluation/clinical_cost_plot.py --objective hitl_cost_weighted_acc --human-acc 0.90
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
        "  .\\src\\.venv\\Scripts\\python.exe evaluation/clinical_cost_plot.py\n"
        "Or install deps:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_BUDGETS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]


def _read_current_tau() -> float:
    with open(ARTIFACTS_DIR / "step3_best_threshold.json") as f:
        best = json.load(f)
    return float(best["best_threshold"])


def _objective_key(objective: str) -> tuple[str, bool]:
    """
    Return (column_name, maximize?).

    - cost_reduction: maximize
    - hitl_cost_weighted_acc: maximize
    - hitl_cost: minimize
    """
    obj = objective.strip().lower()
    if obj in {"cost_reduction", "reduction"}:
        return "cost_reduction", True
    if obj in {"hitl_cost_weighted_acc", "cwa", "cost_weighted_acc"}:
        return "hitl_cost_weighted_acc", True
    if obj in {"hitl_cost", "cost"}:
        return "hitl_cost", False
    raise ValueError(f"Unknown objective: {objective}")


def budget_summary(cost_df: pd.DataFrame, human_acc: float, budgets: list[float], objective: str) -> pd.DataFrame:
    col, maximize = _objective_key(objective)
    sub = cost_df[cost_df["human_accuracy"] == human_acc].copy()
    if sub.empty:
        raise ValueError(f"No rows in cost_analysis.csv for human_accuracy={human_acc}")

    rows: list[dict] = []
    for b in budgets:
        feasible = sub[sub["review_rate"] <= b]
        if feasible.empty:
            continue
        idx = feasible[col].idxmax() if maximize else feasible[col].idxmin()
        best = feasible.loc[idx]
        rows.append(
            {
                "budget": float(b),
                "best_threshold": round(float(best["threshold"]), 3),
                "review_rate": round(float(best["review_rate"]), 3),
                "ai_cost": round(float(best["ai_cost"]), 3),
                "hitl_cost": round(float(best["hitl_cost"]), 3),
                "cost_reduction": round(float(best["cost_reduction"]), 3),
                "ai_cost_weighted_acc_pct": round(float(best["ai_cost_weighted_acc"]) * 100, 2),
                "hitl_cost_weighted_acc_pct": round(float(best["hitl_cost_weighted_acc"]) * 100, 2),
                "objective": col,
                "objective_value": round(float(best[col]), 6),
            }
        )
    return pd.DataFrame(rows)


def operating_point_summary(cost_df: pd.DataFrame, tau: float, human_accs: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    for h in human_accs:
        sub = cost_df[cost_df["human_accuracy"] == h].copy()
        if sub.empty:
            continue
        idx = (sub["threshold"] - tau).abs().idxmin()
        r = sub.loc[idx]
        rows.append(
            {
                "human_accuracy": float(h),
                "tau_used": float(r["threshold"]),
                "review_rate": float(r["review_rate"]),
                "ai_total_cost": float(r["ai_cost"]),
                "hitl_total_cost": float(r["hitl_cost"]),
                "cost_reduction": float(r["cost_reduction"]),
                "ai_cost_weighted_acc_pct": round(float(r["ai_cost_weighted_acc"]) * 100, 3),
                "hitl_cost_weighted_acc_pct": round(float(r["hitl_cost_weighted_acc"]) * 100, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("human_accuracy")


def plot_tradeoff(
    cost_df: pd.DataFrame,
    current_tau: float,
    human_accs: list[float],
    budget_table: pd.DataFrame | None,
    focus_human_acc: float,
) -> Path | None:
    if plt is None:
        print("matplotlib is not available; skipped plot generation.")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Clinical Cost Trade-off vs Review Workload", fontsize=13, fontweight="bold")

    # Panel 1: cost reduction vs review rate
    ax = axes[0]
    for h in human_accs:
        sub = cost_df[cost_df["human_accuracy"] == h].sort_values("review_rate")
        ax.plot(sub["review_rate"] * 100, sub["cost_reduction"], linewidth=2, label=f"h_acc={h:g}")

    ax.set_xlabel("Review Rate (% cases escalated)", fontsize=11)
    ax.set_ylabel("Cost Reduction vs AI-only (units)", fontsize=11)
    ax.set_title("Cost Reduction vs Workload", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Mark current operating point for focus_human_acc
    sub_focus = cost_df[cost_df["human_accuracy"] == focus_human_acc].copy()
    if not sub_focus.empty:
        idx = (sub_focus["threshold"] - current_tau).abs().idxmin()
        r = sub_focus.loc[idx]
        ax.scatter([float(r["review_rate"]) * 100], [float(r["cost_reduction"])], color="darkorange", s=100, zorder=6)
        ax.annotate(
            f"current tau~{float(r['threshold']):.3f}",
            xy=(float(r["review_rate"]) * 100, float(r["cost_reduction"])),
            xytext=(float(r["review_rate"]) * 100 + 3, float(r["cost_reduction"]) - 8),
            fontsize=8,
            color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
        )

    # Panel 2: cost-weighted accuracy vs review rate
    ax2 = axes[1]
    for h in human_accs:
        sub = cost_df[cost_df["human_accuracy"] == h].sort_values("review_rate")
        ax2.plot(sub["review_rate"] * 100, sub["hitl_cost_weighted_acc"] * 100, linewidth=2, label=f"HITL h_acc={h:g}")

    ai_baseline = float(cost_df["ai_cost_weighted_acc"].iloc[0]) * 100
    ax2.axhline(ai_baseline, color="crimson", linestyle="--", linewidth=1.6, label=f"AI-only ({ai_baseline:.1f}%)")
    ax2.set_xlabel("Review Rate (% cases escalated)", fontsize=11)
    ax2.set_ylabel("Cost-Weighted Accuracy (%)", fontsize=11)
    ax2.set_title("Cost-Weighted Accuracy vs Workload", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # Budget summary markers (focus_human_acc)
    if budget_table is not None and not budget_table.empty:
        for _, row in budget_table.iterrows():
            if float(row["budget"]) in {0.08, 0.20, 0.40}:
                ax.scatter([float(row["review_rate"]) * 100], [float(row["cost_reduction"])], color="purple", marker="D", s=60, zorder=5, alpha=0.75)
                ax2.scatter([float(row["review_rate"]) * 100], [float(row["hitl_cost_weighted_acc_pct"])], color="purple", marker="D", s=60, zorder=5, alpha=0.75)

    plt.tight_layout()
    out = FIGURES_DIR / "clinical_cost_tradeoff.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="cost_reduction", help="Objective: cost_reduction | hitl_cost_weighted_acc | hitl_cost")
    parser.add_argument("--human-acc", type=float, default=0.90, help="Human accuracy to use for budget table (default: 0.90).")
    parser.add_argument("--human-accs", type=float, nargs="*", default=[0.85, 0.90, 0.95], help="Human accuracies to plot (default: 0.85 0.90 0.95).")
    parser.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS, help="Budget levels to summarize.")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cost_df = pd.read_csv(TABLES_DIR / "cost_analysis.csv")
    current_tau = _read_current_tau()

    # Budget table for focus human accuracy
    budget_tbl = budget_summary(
        cost_df=cost_df,
        human_acc=float(args.human_acc),
        budgets=[float(b) for b in args.budgets],
        objective=str(args.objective),
    )
    out_budget = TABLES_DIR / "clinical_cost_budget_summary.csv"
    budget_tbl.to_csv(out_budget, index=False)
    print("\n=== Best Threshold per Budget Level (Clinical Cost) ===")
    print(budget_tbl.to_string(index=False))
    print(f"Saved: {out_budget}")

    # Operating point summary across human accuracy sensitivities
    op_tbl = operating_point_summary(cost_df, current_tau, [float(x) for x in args.human_accs])
    out_op = TABLES_DIR / "clinical_cost_summary.csv"
    op_tbl.to_csv(out_op, index=False)
    print(f"\nSaved: {out_op}")

    plot_tradeoff(
        cost_df=cost_df,
        current_tau=current_tau,
        human_accs=[float(x) for x in args.human_accs],
        budget_table=budget_tbl,
        focus_human_acc=float(args.human_acc),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

