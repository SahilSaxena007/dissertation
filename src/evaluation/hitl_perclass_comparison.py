# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Per-class HITL vs AI-only analysis, plus "best tau per budget" summaries.

This script answers: at a given tau (or review budget), where does the overall
accuracy gain come from (SCD vs MCI vs AD)?

Two modes:
  1) Operating point (default): per-class AI-only vs HITL at current tau*
  2) Budget summary: pick best tau per budget (like threshold_tradeoff_plot.py),
     then report per-class expected accuracies at those taus.

Expected (deterministic) HITL accuracy is computed without Monte Carlo:
  - If escalated: P(correct) = human_acc
  - If not escalated: P(correct) = ai_correct (0/1 per case)

Outputs:
  - reports/tables/hitl_perclass_comparison.csv
  - reports/figures/hitl_perclass_comparison.png (if matplotlib available)
  - reports/tables/hitl_perclass_budget_summary.csv (when --budget-summary)
  - reports/figures/hitl_perclass_budget_comparison.png (when --budget-summary)

Usage (from src/):
  .\\.venv\\Scripts\\python.exe evaluation/hitl_perclass_comparison.py
  .\\.venv\\Scripts\\python.exe evaluation/hitl_perclass_comparison.py --tau 0.55 --human-acc 0.90
  .\\.venv\\Scripts\\python.exe evaluation/hitl_perclass_comparison.py --budget-summary --human-acc 0.90
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
        "  .\\src\\.venv\\Scripts\\python.exe evaluation/hitl_perclass_comparison.py\n"
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

CLASS_COLORS = {"SCD": "#4C9BE8", "MCI": "#E8854C", "AD": "#6DBE6D"}
CLASS_ORDER = ["SCD", "MCI", "AD"]


def _read_operating_point() -> tuple[float, float]:
    with open(ARTIFACTS_DIR / "step3_best_threshold.json") as f:
        best = json.load(f)
    return float(best["best_threshold"]), float(best.get("human_accuracy_assumed", 0.9))


def _load_label_mapping() -> dict[int, str]:
    """
    Robust numeric->name mapping inferred from reports/tables/escalation_table_oof.csv.
    Falls back to {0:SCD,1:MCI,2:AD}.
    """
    fallback = {0: "SCD", 1: "MCI", 2: "AD"}
    path = TABLES_DIR / "escalation_table_oof.csv"
    if not path.exists():
        return fallback

    df = pd.read_csv(path, usecols=["true_label", "true_label_name"]).dropna()
    if df.empty:
        return fallback

    mapping: dict[int, str] = {}
    for label, series in df.groupby("true_label")["true_label_name"]:
        mapping[int(label)] = str(series.value_counts().idxmax())
    return {**fallback, **mapping}


def expected_hitl_correct(ai_correct: np.ndarray, risk: np.ndarray, tau: float, human_acc: float) -> np.ndarray:
    escalated = risk >= tau
    return np.where(escalated, float(human_acc), ai_correct.astype(float))


def per_class_table(
    labels: np.ndarray,
    ai_correct: np.ndarray,
    hitl_correct: np.ndarray,
    risk: np.ndarray,
    tau: float,
    id_to_name: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for class_name in CLASS_ORDER:
        # find numeric id for this class_name
        ids = [k for k, v in id_to_name.items() if v == class_name]
        if not ids:
            continue
        cls_id = int(ids[0])
        mask = labels == cls_id
        n = int(mask.sum())
        if n == 0:
            continue
        ai_acc = float(ai_correct[mask].mean())
        hitl_acc = float(hitl_correct[mask].mean())
        rows.append(
            {
                "class": class_name,
                "n": n,
                "ai_accuracy": round(ai_acc, 6),
                "hitl_accuracy": round(hitl_acc, 6),
                "gain_pp": round((hitl_acc - ai_acc) * 100.0, 3),
                "escalation_rate": round(float((risk[mask] >= tau).mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def budget_summary(
    labels: np.ndarray,
    ai_correct: np.ndarray,
    risk: np.ndarray,
    human_acc: float,
    budgets: list[float],
    id_to_name: dict[int, str],
) -> pd.DataFrame:
    taus = np.unique(np.round(np.linspace(0.0, 1.0, 201), 3))

    best_rows: list[dict] = []
    for b in budgets:
        best = None
        for tau in taus:
            escalated = risk >= tau
            rr = float(escalated.mean())
            if rr > b:
                continue
            hitl_correct = expected_hitl_correct(ai_correct, risk, tau, human_acc)
            policy_acc = float(hitl_correct.mean())
            if (best is None) or (policy_acc > best["expected_policy_accuracy"]):
                best = {
                    "budget": float(b),
                    "best_threshold": float(tau),
                    "review_rate": rr,
                    "expected_policy_accuracy": policy_acc,
                }
        if best is None:
            continue

        tau_star = float(best["best_threshold"])
        hitl_correct_star = expected_hitl_correct(ai_correct, risk, tau_star, human_acc)
        per_cls = per_class_table(labels, ai_correct, hitl_correct_star, risk, tau_star, id_to_name)

        row = {
            "budget": round(float(best["budget"]), 3),
            "best_threshold": round(tau_star, 3),
            "review_rate": round(float(best["review_rate"]), 3),
            "expected_policy_accuracy": round(float(best["expected_policy_accuracy"]), 6),
            "ai_only_accuracy": round(float(ai_correct.mean()), 6),
            "expected_accuracy_gain_pp": round((float(best["expected_policy_accuracy"]) - float(ai_correct.mean())) * 100.0, 2),
        }
        for _, r in per_cls.iterrows():
            cls = str(r["class"]).lower()
            row[f"{cls}_hitl_acc"] = round(float(r["hitl_accuracy"]), 6)
            row[f"{cls}_ai_acc"] = round(float(r["ai_accuracy"]), 6)
            row[f"{cls}_gain_pp"] = round(float(r["gain_pp"]), 3)
            row[f"{cls}_escalation_rate"] = round(float(r["escalation_rate"]), 6)
        best_rows.append(row)

    return pd.DataFrame(best_rows)


def plot_operating_point(per_cls: pd.DataFrame, tau: float, human_acc: float) -> Path | None:
    if plt is None:
        print("matplotlib is not available; skipped plot generation.")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(per_cls))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Per-Class Accuracy: AI-only vs HITL (tau={tau:.3f}, human_acc={human_acc:.2f})", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(x - width / 2, per_cls["ai_accuracy"] * 100, width, label="AI-only", color="#E05C5C", alpha=0.85)
    ax.bar(x + width / 2, per_cls["hitl_accuracy"] * 100, width, label="HITL", color="#4C9BE8", alpha=0.85)
    for i, row in per_cls.iterrows():
        gain = float(row["gain_pp"])
        ax.annotate(
            f"+{gain:.1f}pp" if gain >= 0 else f"{gain:.1f}pp",
            xy=(i + width / 2, float(row["hitl_accuracy"]) * 100 + 0.5),
            ha="center",
            va="bottom",
            fontsize=9,
            color="green" if gain >= 0 else "red",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(per_cls["class"], per_cls["n"])], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy by Diagnosis Class", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colors = [CLASS_COLORS.get(str(c), "#999999") for c in per_cls["class"].tolist()]
    ax2.bar(x, per_cls["gain_pp"], color=colors, alpha=0.85, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2b = ax2.twinx()
    ax2b.plot(x, per_cls["escalation_rate"] * 100, "D--", color="purple", linewidth=1.8, markersize=8, label="Escalation rate")
    ax2b.set_ylabel("Escalation Rate (%)", fontsize=10, color="purple")
    ax2b.tick_params(axis="y", labelcolor="purple")
    ax2b.set_ylim(0, 100)
    ax2b.legend(loc="upper right", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(per_cls["class"].tolist(), fontsize=11)
    ax2.set_ylabel("Accuracy Gain (pp)", fontsize=11)
    ax2.set_title("Gain per Class vs Escalation Rate", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_fig = FIGURES_DIR / "hitl_perclass_comparison.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")
    return out_fig


def plot_budget_comparison(budget_tbl: pd.DataFrame) -> Path | None:
    if plt is None:
        print("matplotlib is not available; skipped plot generation.")
        return None
    if budget_tbl.empty:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only a few canonical budgets for readability.
    keep = budget_tbl[budget_tbl["budget"].isin([0.08, 0.20, 0.40])]
    if keep.empty:
        keep = budget_tbl.head(3)

    classes = ["scd", "mci", "ad"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title("Per-Class HITL Accuracy at Budget-Optimal Operating Points", fontsize=12, fontweight="bold")

    x = np.arange(len(classes))
    width = 0.18

    # Baseline AI-only per class at each budget-optimal tau (changes slightly due to tau-dependent class mix of non-escalated cases).
    for j, (_, row) in enumerate(keep.iterrows()):
        label = f"{int(float(row['budget']) * 100)}% (tau={float(row['best_threshold']):.3f})"
        hitl_vals = [float(row.get(f"{c}_hitl_acc", np.nan)) * 100 for c in classes]
        ax.bar(x + (j - 1) * width, hitl_vals, width=width, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes], fontsize=11)
    ax.set_ylabel("HITL Expected Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "hitl_perclass_budget_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=None, help="Override tau for operating-point comparison.")
    parser.add_argument("--human-acc", type=float, default=None, help="Override clinician accuracy for expected HITL accuracy.")
    parser.add_argument("--budget-summary", action="store_true", help="Print/save best tau per budget with per-class stats.")
    parser.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS, help="Budget levels to summarize.")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tau_default, human_acc_default = _read_operating_point()
    tau = float(args.tau) if args.tau is not None else tau_default
    human_acc = float(args.human_acc) if args.human_acc is not None else human_acc_default

    df = pd.read_csv(TABLES_DIR / "threshold_data.csv")
    labels = df["true_label"].to_numpy(dtype=int)
    ai_correct = df["ai_correct"].astype(bool).to_numpy(dtype=bool).astype(float)
    risk = df["review_risk_score"].to_numpy(dtype=float)

    id_to_name = _load_label_mapping()

    # Operating point outputs
    hitl_correct = expected_hitl_correct(ai_correct, risk, tau, human_acc)
    per_cls = per_class_table(labels, ai_correct, hitl_correct, risk, tau, id_to_name)

    out_csv = TABLES_DIR / "hitl_perclass_comparison.csv"
    per_cls.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(per_cls.to_string(index=False))
    plot_operating_point(per_cls, tau, human_acc)

    if args.budget_summary:
        budget_tbl = budget_summary(
            labels=labels,
            ai_correct=ai_correct,
            risk=risk,
            human_acc=human_acc,
            budgets=[float(b) for b in args.budgets],
            id_to_name=id_to_name,
        )
        out_budget = TABLES_DIR / "hitl_perclass_budget_summary.csv"
        budget_tbl.to_csv(out_budget, index=False)
        print("\n=== Best Threshold per Budget Level (Per-Class Expected Accuracy) ===")
        print(budget_tbl.to_string(index=False))
        print(f"Saved: {out_budget}")
        plot_budget_comparison(budget_tbl)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

