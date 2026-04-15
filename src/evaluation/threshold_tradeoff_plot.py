# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Threshold trade-off analysis: accuracy gain vs review rate.

Reads the full threshold sweep from threshold_analysis.csv and produces:
  1. A plot of expected accuracy gain vs review rate (with CI band)
  2. A summary table of best threshold at common budget levels
  3. Marks the current operating point and the unconstrained optimum

Usage (from src/):
    python evaluation/threshold_tradeoff_plot.py
    python evaluation/threshold_tradeoff_plot.py --update-threshold 0.55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_data() -> pd.DataFrame:
    path = TABLES_DIR / "threshold_analysis.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"threshold_analysis.csv not found at {path}. "
            "Run escalation/threshold_runner.py first."
        )
    return pd.read_csv(path)


def budget_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Best threshold (max expected accuracy gain) at each budget level."""
    budgets = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]
    rows = []
    for b in budgets:
        feasible = df[df["review_rate"] <= b]
        if feasible.empty:
            continue
        best = feasible.loc[feasible["expected_accuracy_gain"].idxmax()]
        rows.append({
            "budget": b,
            "best_threshold": round(float(best["threshold"]), 3),
            "review_rate": round(float(best["review_rate"]), 3),
            "expected_accuracy_gain_pp": round(float(best["expected_accuracy_gain"]) * 100, 2),
            "expected_policy_accuracy": round(float(best["expected_policy_accuracy"]), 4),
            "min_policy_accuracy": round(float(best["min_policy_accuracy"]), 4),
            "max_policy_accuracy": round(float(best["max_policy_accuracy"]), 4),
        })
    return pd.DataFrame(rows)


def plot_tradeoff(df: pd.DataFrame, current_tau: float, summary: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ai_only = float(df["ai_only_accuracy"].iloc[0])

    # Sort by review_rate for clean line
    df_plot = df.sort_values("review_rate").copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("HITL Escalation Threshold: Accuracy–Workload Trade-off", fontsize=14, fontweight="bold")

    # ── Left panel: Accuracy gain vs review rate ──────────────────────────
    ax = axes[0]
    ax.fill_between(
        df_plot["review_rate"] * 100,
        df_plot["min_policy_accuracy"] * 100,
        df_plot["max_policy_accuracy"] * 100,
        alpha=0.15, color="steelblue", label="Policy accuracy range\n(h_acc ∈ [0.85, 0.95])"
    )
    ax.plot(
        df_plot["review_rate"] * 100,
        df_plot["expected_policy_accuracy"] * 100,
        color="steelblue", linewidth=2.5, label="Expected policy accuracy"
    )
    ax.axhline(ai_only * 100, color="crimson", linestyle="--", linewidth=1.8,
               label=f"AI-only baseline ({ai_only*100:.1f}%)")

    # Mark current operating point
    cur = df[np.isclose(df["threshold"], current_tau, atol=0.003)]
    if not cur.empty:
        cr = float(cur["review_rate"].iloc[0])
        ca = float(cur["expected_policy_accuracy"].iloc[0])
        ax.scatter([cr * 100], [ca * 100], color="darkorange", s=120, zorder=5,
                   label=f"Current τ*={current_tau} ({cr*100:.1f}% reviewed)")
        ax.annotate(f"τ*={current_tau}\n+{(ca-ai_only)*100:.1f}pp",
                    xy=(cr * 100, ca * 100), xytext=(cr * 100 + 4, ca * 100 - 1.5),
                    fontsize=8, color="darkorange",
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

    # Mark unconstrained optimum
    opt_idx = df_plot["expected_accuracy_gain"].idxmax()
    opt = df_plot.loc[opt_idx]
    ax.scatter([opt["review_rate"] * 100], [opt["expected_policy_accuracy"] * 100],
               color="green", marker="*", s=180, zorder=5,
               label=f"Unconstrained optimum\n({opt['review_rate']*100:.1f}% reviewed, +{opt['expected_accuracy_gain']*100:.1f}pp)")

    # Mark budget summary points
    for _, row in summary.iterrows():
        if row["budget"] in [0.08, 0.20, 0.40]:
            ax.scatter([row["review_rate"] * 100], [row["expected_policy_accuracy"] * 100],
                       color="purple", marker="D", s=60, zorder=4, alpha=0.7)

    ax.set_xlabel("Review Rate (% of cases escalated to clinician)", fontsize=11)
    ax.set_ylabel("Policy Accuracy (%)", fontsize=11)
    ax.set_title("Policy Accuracy vs Clinician Workload", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 101)

    # ── Right panel: Accuracy gain (pp) vs review rate ───────────────────
    ax2 = axes[1]
    ax2.fill_between(
        df_plot["review_rate"] * 100,
        (df_plot["min_policy_accuracy"] - ai_only) * 100,
        (df_plot["max_policy_accuracy"] - ai_only) * 100,
        alpha=0.15, color="steelblue"
    )
    ax2.plot(
        df_plot["review_rate"] * 100,
        df_plot["expected_accuracy_gain"] * 100,
        color="steelblue", linewidth=2.5
    )
    ax2.axhline(0, color="crimson", linestyle="--", linewidth=1.8, label="No gain threshold")

    # Budget operating points from summary table
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(summary)))
    for i, (_, row) in enumerate(summary.iterrows()):
        ax2.scatter([row["review_rate"] * 100], [row["expected_accuracy_gain_pp"]],
                    color=colors[i], s=80, zorder=5)
        if row["budget"] in [0.08, 0.20, 0.40]:
            ax2.annotate(
                f"{int(row['budget']*100)}% budget\n+{row['expected_accuracy_gain_pp']:.1f}pp",
                xy=(row["review_rate"] * 100, row["expected_accuracy_gain_pp"]),
                xytext=(row["review_rate"] * 100 + 2, row["expected_accuracy_gain_pp"] + 0.3),
                fontsize=8, color=colors[i]
            )

    ax2.set_xlabel("Review Rate (% of cases escalated to clinician)", fontsize=11)
    ax2.set_ylabel("Accuracy Gain over AI-only (percentage points)", fontsize=11)
    ax2.set_title("Accuracy Gain vs Clinician Workload", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-1, 101)

    plt.tight_layout()
    out = FIGURES_DIR / "threshold_tradeoff.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out}")
    return out


def update_threshold(new_tau: float, df: pd.DataFrame) -> None:
    """Rewrite step3_best_threshold.json with a new operating point."""
    row = df[np.isclose(df["threshold"], new_tau, atol=0.003)]
    if row.empty:
        # Find nearest
        idx = (df["threshold"] - new_tau).abs().idxmin()
        row = df.iloc[[idx]]
        print(f"Exact τ={new_tau} not found, using nearest: {row['threshold'].iloc[0]:.3f}")

    r = row.iloc[0]
    ai_only = float(df["ai_only_accuracy"].iloc[0])

    # Load existing file to preserve fields we don't recompute
    existing_path = ARTIFACTS_DIR / "step3_best_threshold.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

    updated = {
        **existing,
        "best_threshold": round(float(r["threshold"]), 3),
        "review_rate": round(float(r["review_rate"]), 4),
        "policy_accuracy": round(float(r["expected_policy_accuracy"]), 6),
        "ai_only_accuracy": round(ai_only, 6),
        "accuracy_gain": round(float(r["expected_accuracy_gain"]), 6),
        "worst_case_policy_accuracy": round(float(r["min_policy_accuracy"]), 6),
        "best_case_policy_accuracy": round(float(r["max_policy_accuracy"]), 6),
    }

    with open(existing_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated step3_best_threshold.json -> tau={updated['best_threshold']}, "
          f"review_rate={updated['review_rate']:.1%}, "
          f"accuracy_gain=+{updated['accuracy_gain']*100:.2f}pp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-threshold", type=float, default=None,
        help="If provided, rewrite step3_best_threshold.json with this τ value."
    )
    args = parser.parse_args()

    df = load_data()

    # Current operating point from existing JSON
    json_path = ARTIFACTS_DIR / "step3_best_threshold.json"
    current_tau = 0.765
    if json_path.exists():
        with open(json_path) as f:
            current_tau = float(json.load(f).get("best_threshold", 0.765))

    summary = budget_summary(df)
    print("\n=== Best Threshold per Budget Level ===")
    print(summary.to_string(index=False))

    plot_tradeoff(df, current_tau, summary)

    if args.update_threshold is not None:
        update_threshold(args.update_threshold, df)


if __name__ == "__main__":
    main()
