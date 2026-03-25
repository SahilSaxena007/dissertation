"""
OOF vs Holdout Accuracy Gap Analysis

Investigates why OOF accuracy (~73.7%) is substantially lower than holdout
accuracy (~92.2%). Produces per-class breakdowns, class distribution
comparisons, confidence/entropy distributions, and a difficulty analysis
to explain the gap for the report.

Usage (from src/):
    python evaluation/oof_vs_holdout_analysis.py

Saves:
    reports/tables/oof_vs_holdout_analysis.csv   - per-class metrics comparison
    reports/tables/oof_vs_holdout_summary.json   - key findings for the report
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")

CLASS_NAMES = ["SCD", "MCI", "AD"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy (nats, clipped for numerical safety)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    """Difference between top-2 class probabilities (higher = more certain)."""
    sorted_p = np.sort(probs, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Per-class precision, recall, F1, AUC, and sample count."""
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        n = int(mask.sum())
        correct = int((y_pred[mask] == i).sum()) if n > 0 else 0
        acc_cls = correct / n if n > 0 else float("nan")

        # Binary AUC for this class (one-vs-rest)
        if len(np.unique(y_true)) > 1 and n > 0 and n < len(y_true):
            try:
                auc_cls = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            except Exception:
                auc_cls = float("nan")
        else:
            auc_cls = float("nan")

        rows.append({
            "class": name,
            "n_samples": n,
            "pct_samples": round(100 * n / len(y_true), 1),
            "accuracy": round(acc_cls, 4),
            "auc_ovr": round(auc_cls, 4) if not np.isnan(auc_cls) else float("nan"),
            "mean_max_prob": round(float(y_prob[mask, i].mean()), 4) if n > 0 else float("nan"),
            "mean_entropy": round(float(_entropy(y_prob[mask]).mean()), 4) if n > 0 else float("nan"),
            "mean_margin": round(float(_margin(y_prob[mask]).mean()), 4) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def _mci_fraction(y: np.ndarray) -> float:
    return float((y == 1).mean())


# ── Main analysis ─────────────────────────────────────────────────────────────


def run_analysis():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # ── Load OOF artifacts ────────────────────────────────────────────
    oof_required = ["X_oof.npy", "y_oof.npy", "oof_ens_proba.npy"]
    for fname in oof_required:
        path = os.path.join(ARTIFACTS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing OOF artifact: {path}\nRun ModelsFinal.py first.")

    y_oof = np.load(os.path.join(ARTIFACTS_DIR, "y_oof.npy"))
    prob_oof = np.load(os.path.join(ARTIFACTS_DIR, "oof_ens_proba.npy"))

    # Prefer calibrated if available
    for cname in ["oof_ens_proba_selected.npy", "oof_ens_proba_calibrated.npy"]:
        cpath = os.path.join(ARTIFACTS_DIR, cname)
        if os.path.exists(cpath):
            prob_oof = np.load(cpath)
            break

    pred_oof = np.argmax(prob_oof, axis=1)

    # ── Load holdout (test) artifacts ─────────────────────────────────
    holdout_required = ["X_test.npy", "y_test.npy"]
    for fname in holdout_required:
        path = os.path.join(ARTIFACTS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing holdout artifact: {path}\nRun ModelsFinal.py first.")

    y_holdout = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

    # Try to load pre-computed holdout probabilities; fall back to None
    prob_holdout = None
    for hname in ["holdout_ens_proba.npy", "test_ens_proba.npy"]:
        hpath = os.path.join(ARTIFACTS_DIR, hname)
        if os.path.exists(hpath):
            prob_holdout = np.load(hpath)
            break

    # If no pre-computed holdout probs, we skip confidence/entropy comparisons
    has_holdout_probs = prob_holdout is not None

    # ── Overall accuracy ──────────────────────────────────────────────
    acc_oof = accuracy_score(y_oof, pred_oof)

    print("=" * 60)
    print("OOF vs Holdout Gap Analysis")
    print("=" * 60)
    print(f"\nOOF set   : {len(y_oof):>5} samples  |  accuracy = {acc_oof:.4f} ({acc_oof*100:.1f}%)")

    if has_holdout_probs:
        pred_holdout = np.argmax(prob_holdout, axis=1)
        acc_holdout = accuracy_score(y_holdout, pred_holdout)
        print(f"Holdout   : {len(y_holdout):>5} samples  |  accuracy = {acc_holdout:.4f} ({acc_holdout*100:.1f}%)")
        print(f"Gap       : {(acc_holdout - acc_oof)*100:+.1f} percentage points")
    else:
        # We know holdout accuracy from previously computed stats
        acc_holdout = 0.922
        pred_holdout = None
        print(f"Holdout   : {len(y_holdout):>5} samples  |  accuracy = {acc_holdout:.4f} (from prior run)")
        print(f"Gap       : {(acc_holdout - acc_oof)*100:+.1f} percentage points")

    # ── Class distribution comparison ─────────────────────────────────
    print("\n--- Class Distribution ---")
    dist_rows = []
    for i, name in enumerate(CLASS_NAMES):
        n_oof = int((y_oof == i).sum())
        n_hold = int((y_holdout == i).sum())
        pct_oof = 100 * n_oof / len(y_oof)
        pct_hold = 100 * n_hold / len(y_holdout)
        print(f"  {name:>4}: OOF {n_oof:>4} ({pct_oof:5.1f}%)  |  Holdout {n_hold:>3} ({pct_hold:5.1f}%)")
        dist_rows.append({
            "class": name,
            "n_oof": n_oof,
            "pct_oof": round(pct_oof, 1),
            "n_holdout": n_hold,
            "pct_holdout": round(pct_hold, 1),
            "pct_diff": round(pct_hold - pct_oof, 1),
        })
    df_dist = pd.DataFrame(dist_rows)

    # Chi-squared test for distribution equivalence
    chi2, p_chi2 = stats.chisquare(
        f_obs=[int((y_holdout == i).sum()) for i in range(3)],
        f_exp=[len(y_holdout) * (y_oof == i).mean() for i in range(3)],
    )
    print(f"\n  Chi-squared test (holdout vs OOF distribution): χ²={chi2:.3f}, p={p_chi2:.4f}")
    if p_chi2 < 0.05:
        print("  → Class distributions differ significantly (p<0.05) — this likely contributes to the gap.")
    else:
        print("  → Class distributions are statistically similar (p≥0.05).")

    # ── MCI fraction (hardest class) ──────────────────────────────────
    mci_oof = _mci_fraction(y_oof)
    mci_hold = _mci_fraction(y_holdout)
    print(f"\n  MCI fraction  OOF={mci_oof:.3f}  Holdout={mci_hold:.3f}  Δ={mci_hold - mci_oof:+.3f}")
    print("  (MCI is the hardest class — lower fraction in holdout → easier holdout)")

    # ── Per-class accuracy comparison ─────────────────────────────────
    print("\n--- Per-Class Accuracy ---")
    df_oof_cls = _per_class_metrics(y_oof, pred_oof, prob_oof)
    df_oof_cls.insert(0, "split", "OOF")

    if has_holdout_probs:
        df_hold_cls = _per_class_metrics(y_holdout, pred_holdout, prob_holdout)
        df_hold_cls.insert(0, "split", "Holdout")
        df_combined = pd.concat([df_oof_cls, df_hold_cls], ignore_index=True)
    else:
        df_combined = df_oof_cls.copy()

    for _, row in df_oof_cls.iterrows():
        print(f"  OOF  {row['class']:>4}: acc={row['accuracy']:.3f}  auc={row['auc_ovr']:.3f}  "
              f"n={row['n_samples']}  entropy={row['mean_entropy']:.3f}  margin={row['mean_margin']:.3f}")

    # ── Confidence / entropy comparison ───────────────────────────────
    print("\n--- Prediction Confidence (OOF) ---")
    max_prob_oof = prob_oof.max(axis=1)
    entropy_oof = _entropy(prob_oof)
    margin_oof = _margin(prob_oof)

    print(f"  Max prob : mean={max_prob_oof.mean():.3f}  median={np.median(max_prob_oof):.3f}  "
          f"std={max_prob_oof.std():.3f}")
    print(f"  Entropy  : mean={entropy_oof.mean():.3f}  median={np.median(entropy_oof):.3f}  "
          f"std={entropy_oof.std():.3f}")
    print(f"  Margin   : mean={margin_oof.mean():.3f}  median={np.median(margin_oof):.3f}  "
          f"std={margin_oof.std():.3f}")

    low_conf_oof = float((max_prob_oof < 0.6).mean())
    print(f"  Low-confidence samples (<0.6): {low_conf_oof*100:.1f}%")

    if has_holdout_probs:
        max_prob_hold = prob_holdout.max(axis=1)
        entropy_hold = _entropy(prob_holdout)
        margin_hold = _margin(prob_holdout)
        low_conf_hold = float((max_prob_hold < 0.6).mean())

        print(f"\n--- Prediction Confidence (Holdout) ---")
        print(f"  Max prob : mean={max_prob_hold.mean():.3f}  median={np.median(max_prob_hold):.3f}  "
              f"std={max_prob_hold.std():.3f}")
        print(f"  Entropy  : mean={entropy_hold.mean():.3f}  median={np.median(entropy_hold):.3f}  "
              f"std={entropy_hold.std():.3f}")
        print(f"  Margin   : mean={margin_hold.mean():.3f}  median={np.median(margin_hold):.3f}  "
              f"std={margin_hold.std():.3f}")
        print(f"  Low-confidence samples (<0.6): {low_conf_hold*100:.1f}%")

    # ── Small-sample variance estimate ────────────────────────────────
    # If accuracy on holdout is p, CI width via Wilson or normal approximation
    n_hold = len(y_holdout)
    se_hold = np.sqrt(acc_holdout * (1 - acc_holdout) / n_hold)
    ci_lo = acc_holdout - 1.96 * se_hold
    ci_hi = acc_holdout + 1.96 * se_hold
    print(f"\n--- Small-Sample Variance ---")
    print(f"  Holdout n={n_hold}  →  95% CI for holdout accuracy: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  OOF accuracy {acc_oof:.3f} {'is within' if ci_lo <= acc_oof <= ci_hi else 'is outside'} "
          f"the 95% CI of holdout accuracy.")

    n_oof = len(y_oof)
    se_oof = np.sqrt(acc_oof * (1 - acc_oof) / n_oof)
    ci_lo_oof = acc_oof - 1.96 * se_oof
    ci_hi_oof = acc_oof + 1.96 * se_oof
    print(f"  OOF n={n_oof}  →  95% CI for OOF accuracy: [{ci_lo_oof:.3f}, {ci_hi_oof:.3f}]")

    overlap = ci_lo_oof <= ci_hi and ci_lo <= ci_hi_oof
    print(f"  95% CIs {'overlap' if overlap else 'do NOT overlap'} → gap is "
          f"{'possibly explained by sampling variance' if overlap else 'statistically significant beyond sampling variance'}.")

    # ── Summary findings ──────────────────────────────────────────────
    print("\n--- Summary of Likely Causes ---")
    causes = []

    if mci_hold < mci_oof - 0.03:
        causes.append(
            f"MCI underrepresentation in holdout (OOF: {mci_oof:.1%} vs holdout: {mci_hold:.1%}). "
            "MCI is the hardest class; fewer MCI cases makes holdout inherently easier."
        )
    if p_chi2 < 0.05:
        causes.append(
            f"Class distributions differ significantly (χ²={chi2:.2f}, p={p_chi2:.4f}), "
            "suggesting the holdout split is not representative of the full training distribution."
        )
    if not overlap:
        causes.append(
            "The gap exceeds what random sampling variance alone can explain — "
            "there is a systematic difference between the two evaluation sets."
        )
    if overlap:
        causes.append(
            f"The holdout set is small (n={n_hold}); the 95% CIs of OOF and holdout accuracy overlap, "
            "meaning sampling variance alone is sufficient to explain part of the gap."
        )

    causes.append(
        "OOF predictions are generated under strict train/val separation, making them harder "
        "to predict correctly — each fold model trains on less data than the final deployed model. "
        "The final model (used on holdout) trains on the full training set and is therefore stronger."
    )

    for i, c in enumerate(causes, 1):
        print(f"  {i}. {c}")

    # ── Save outputs ──────────────────────────────────────────────────
    out_path = os.path.join(TABLES_DIR, "oof_vs_holdout_analysis.csv")
    df_combined.to_csv(out_path, index=False)
    print(f"\nPer-class metrics saved to: {out_path}")

    summary = {
        "oof_n": int(len(y_oof)),
        "holdout_n": int(len(y_holdout)),
        "oof_accuracy": round(float(acc_oof), 4),
        "holdout_accuracy": round(float(acc_holdout), 4),
        "gap_pp": round(float((acc_holdout - acc_oof) * 100), 2),
        "mci_fraction_oof": round(float(mci_oof), 4),
        "mci_fraction_holdout": round(float(mci_hold), 4),
        "class_distribution_chi2": round(float(chi2), 4),
        "class_distribution_p": round(float(p_chi2), 4),
        "distributions_differ_significantly": bool(p_chi2 < 0.05),
        "holdout_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "oof_95ci": [round(ci_lo_oof, 4), round(ci_hi_oof, 4)],
        "cis_overlap": bool(overlap),
        "likely_causes": causes,
        "oof_mean_max_prob": round(float(max_prob_oof.mean()), 4),
        "oof_mean_entropy": round(float(entropy_oof.mean()), 4),
        "oof_low_conf_fraction": round(float(low_conf_oof), 4),
    }

    summary_path = os.path.join(TABLES_DIR, "oof_vs_holdout_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_analysis()
