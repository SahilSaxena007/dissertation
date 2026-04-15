# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Active learning curve: accuracy and macro F1 over feedback rounds.

Reads hitl_simulated_clinician_experiment.csv and plots:
  1. Accuracy over rounds per clinician accuracy level
  2. Macro F1 over rounds per clinician accuracy level
  3. Gain over AI-only baseline per round

Usage (from src/):
    python evaluation/active_learning_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

H_ACC_COLORS = {0.85: "#E8854C", 0.90: "#4C9BE8", 0.95: "#6DBE6D"}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    exp_path = TABLES_DIR / "hitl_simulated_clinician_experiment.csv"
    if not exp_path.exists():
        raise FileNotFoundError(
            f"{exp_path} not found. Run hitl/run_experiment.py first."
        )

    df = pd.read_csv(exp_path)
    print("Loaded experiment results:")
    print(df.to_string(index=False))

    h_accs = sorted(df["clinician_accuracy"].unique())

    # Baseline = round 0 (same for all h_acc levels)
    baseline_acc = float(df[df["round"] == 0]["test_accuracy"].iloc[0])
    baseline_f1 = float(df[df["round"] == 0]["test_macro_f1"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Active Learning: Model Improvement over Clinician Feedback Rounds",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Accuracy over rounds ────────────────────────────────────
    ax = axes[0]
    ax.axhline(baseline_acc * 100, color="crimson", linestyle="--",
               linewidth=1.8, label=f"AI-only baseline ({baseline_acc*100:.1f}%)")

    for h in h_accs:
        sub = df[df["clinician_accuracy"] == h].sort_values("round")
        color = H_ACC_COLORS.get(h, "gray")

        if "test_accuracy_std" in sub.columns:
            ax.fill_between(
                sub["round"],
                (sub["test_accuracy"] - sub["test_accuracy_std"]) * 100,
                (sub["test_accuracy"] + sub["test_accuracy_std"]) * 100,
                alpha=0.15, color=color
            )

        ax.plot(sub["round"], sub["test_accuracy"] * 100,
                color=color, linewidth=2.2, marker="o", markersize=6,
                label=f"h_acc={h}")

    ax.set_xlabel("Feedback Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Test Accuracy vs Round", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2: Macro F1 over rounds ────────────────────────────────────
    ax2 = axes[1]
    ax2.axhline(baseline_f1 * 100, color="crimson", linestyle="--",
                linewidth=1.8, label=f"AI-only baseline ({baseline_f1*100:.1f}%)")

    for h in h_accs:
        sub = df[df["clinician_accuracy"] == h].sort_values("round")
        color = H_ACC_COLORS.get(h, "gray")

        if "test_macro_f1_std" in sub.columns:
            ax2.fill_between(
                sub["round"],
                (sub["test_macro_f1"] - sub["test_macro_f1_std"]) * 100,
                (sub["test_macro_f1"] + sub["test_macro_f1_std"]) * 100,
                alpha=0.15, color=color
            )

        ax2.plot(sub["round"], sub["test_macro_f1"] * 100,
                 color=color, linewidth=2.2, marker="s", markersize=6,
                 label=f"h_acc={h}")

    ax2.set_xlabel("Feedback Round", fontsize=11)
    ax2.set_ylabel("Test Macro F1 (%)", fontsize=11)
    ax2.set_title("Macro F1 vs Round", fontsize=11)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Accuracy gain over baseline per round ───────────────────
    ax3 = axes[2]
    ax3.axhline(0, color="crimson", linestyle="--", linewidth=1.2,
                label="No gain vs AI-only")

    for h in h_accs:
        sub = df[df["clinician_accuracy"] == h].sort_values("round")
        gain = (sub["test_accuracy"] - baseline_acc) * 100
        color = H_ACC_COLORS.get(h, "gray")
        ax3.plot(sub["round"], gain,
                 color=color, linewidth=2.2, marker="^", markersize=6,
                 label=f"h_acc={h}")

        # Annotate final round gain
        final_gain = float(gain.iloc[-1])
        final_round = int(sub["round"].iloc[-1])
        ax3.annotate(
            f"+{final_gain:.1f}pp",
            xy=(final_round, final_gain),
            xytext=(final_round + 0.1, final_gain + 0.1),
            fontsize=8, color=color
        )

    ax3.set_xlabel("Feedback Round", fontsize=11)
    ax3.set_ylabel("Accuracy Gain over AI-only (pp)", fontsize=11)
    ax3.set_title("Cumulative Gain vs Round", fontsize=11)
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "active_learning_curve.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # Summary table
    print("\n=== Gain per round per h_acc ===")
    summary_rows = []
    for h in h_accs:
        sub = df[df["clinician_accuracy"] == h].sort_values("round")
        for _, row in sub.iterrows():
            summary_rows.append({
                "clinician_accuracy": h,
                "round": int(row["round"]),
                "feedback_pool_size": int(row["feedback_pool_size"]),
                "test_accuracy": round(float(row["test_accuracy"]), 4),
                "test_macro_f1": round(float(row["test_macro_f1"]), 4),
                "accuracy_gain_pp": round((float(row["test_accuracy"]) - baseline_acc) * 100, 2),
                "f1_gain_pp": round((float(row["test_macro_f1"]) - baseline_f1) * 100, 2),
            })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(TABLES_DIR / "active_learning_summary.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'active_learning_summary.csv'}")


if __name__ == "__main__":
    main()
