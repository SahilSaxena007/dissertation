"""
Issue #10: Statistical Testing for HITL Pipeline

McNemar's test for paired accuracy, bootstrap AUC-difference test, and bootstrap CIs.
Saves: reports/tables/statistical_tests.csv

Usage (from src/):
    python evaluation/statistical_tests.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test for paired accuracy comparison.
    H0: both models have the same error rate.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency: b wrong & a right (b), a wrong & b right (c)
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # Use continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))

    matched_odds_ratio = float(c / b) if b > 0 else (float("inf") if c > 0 else 1.0)
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "b": b,
        "c": c,
        "matched_odds_ratio_better_b_over_a": matched_odds_ratio,
    }


def bootstrap_auc_diff_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> dict:
    """
    Bootstrap AUC-difference test for multiclass OVR.
    This is NOT true DeLong; naming is explicit to avoid ambiguity.
    """
    auc_a = roc_auc_score(y_true, probs_a, multi_class="ovr")
    auc_b = roc_auc_score(y_true, probs_b, multi_class="ovr")

    rng = np.random.default_rng(42)
    n = len(y_true)
    n_boot = 2000
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            a_b = roc_auc_score(y_b, probs_a[idx], multi_class="ovr")
            b_b = roc_auc_score(y_b, probs_b[idx], multi_class="ovr")
            diffs.append(a_b - b_b)
        except ValueError:
            continue

    diffs = np.array(diffs)
    if len(diffs) < 100:
        return {"auc_a": auc_a, "auc_b": auc_b, "p_value": float("nan"), "note": "insufficient_bootstrap_samples"}

    se = np.std(diffs, ddof=1)
    z = (auc_a - auc_b) / max(se, 1e-12)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(auc_a - auc_b),
        "z_statistic": float(z),
        "p_value": p_value,
        "method": "bootstrap_auc_diff",
    }


def bootstrap_paired_metric_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for paired metric difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        m_a = metric_fn(y_true[idx], y_pred_a[idx])
        m_b = metric_fn(y_true[idx], y_pred_b[idx])
        diffs.append(m_a - m_b)

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        f"ci_lower_{int((1-alpha)*100)}": float(np.percentile(diffs, 100 * alpha / 2)),
        f"ci_upper_{int((1-alpha)*100)}": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        f"ci_lower_{int((1-alpha)*100)}": float(np.percentile(vals, 100 * alpha / 2)),
        f"ci_upper_{int((1-alpha)*100)}": float(np.percentile(vals, 100 * (1 - alpha / 2))),
    }


def run_statistical_tests():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load threshold data (has AI predictions and risk scores)
    thresh_path = os.path.join(TABLES_DIR, "threshold_data.csv")
    df = pd.read_csv(thresh_path)
    y_true = df["true_label"].values.astype(int)
    y_ai = df["pred_ens"].values.astype(int)
    risk_scores = df["review_risk_score"].values.astype(float)

    # Load best threshold
    thresh_info = json.load(open(os.path.join(ARTIFACTS_DIR, "step3_best_threshold.json"), "r"))
    tau = float(thresh_info["best_threshold"])
    h_acc = float(thresh_info.get("human_accuracy_assumed", 0.9))

    # Simulate HITL predictions: escalated cases get simulated clinician correction
    escalated = risk_scores >= tau
    rng = np.random.default_rng(42)
    y_hitl = y_ai.copy()
    esc_idx = np.where(escalated)[0]
    for i in esc_idx:
        if rng.random() < h_acc:
            y_hitl[i] = y_true[i]
        else:
            choices = [c for c in range(3) if c != y_true[i]]
            y_hitl[i] = rng.choice(choices)

    # Load OOF probabilities
    probs_ens = np.load(os.path.join(ARTIFACTS_DIR, "oof_ens_proba_selected.npy"))
    if len(probs_ens) > len(y_true):
        probs_ens = probs_ens[:len(y_true)]

    results = []

    ai_acc = float(accuracy_score(y_true, y_ai))
    hitl_acc = float(accuracy_score(y_true, y_hitl))
    ai_f1 = float(f1_score(y_true, y_ai, average="macro"))
    hitl_f1 = float(f1_score(y_true, y_hitl, average="macro"))

    # 1. McNemar's test
    mcnemar = mcnemar_test(y_true, y_ai, y_hitl)
    results.append({
        "test": "McNemar (AI-only vs HITL)",
        "statistic": mcnemar["statistic"],
        "p_value": mcnemar["p_value"],
        "metric_a": ai_acc,
        "metric_b": hitl_acc,
        "note": (
            f"b={mcnemar['b']}, c={mcnemar['c']}, "
            f"matched_odds_ratio={mcnemar['matched_odds_ratio_better_b_over_a']:.4f}"
        ),
    })

    # 2. Bootstrap CI for accuracy difference
    acc_ci = bootstrap_paired_metric_ci(y_true, y_hitl, y_ai, accuracy_score)
    results.append({
        "test": "Bootstrap CI (HITL - AI accuracy)",
        "statistic": acc_ci["mean_diff"],
        "p_value": float("nan"),
        "metric_a": hitl_acc,
        "metric_b": ai_acc,
        "note": f"95% CI: [{acc_ci['ci_lower_95']:.4f}, {acc_ci['ci_upper_95']:.4f}]",
    })

    f1_ci = bootstrap_paired_metric_ci(
        y_true,
        y_hitl,
        y_ai,
        lambda yt, yp: f1_score(yt, yp, average="macro"),
    )
    results.append({
        "test": "Bootstrap CI (HITL - AI macro F1)",
        "statistic": f1_ci["mean_diff"],
        "p_value": float("nan"),
        "metric_a": hitl_f1,
        "metric_b": ai_f1,
        "note": f"95% CI: [{f1_ci['ci_lower_95']:.4f}, {f1_ci['ci_upper_95']:.4f}]",
    })

    # 3. DeLong-style AUC comparison (AI raw vs calibrated if different)
    probs_raw = np.load(os.path.join(ARTIFACTS_DIR, "oof_ens_proba.npy"))
    if len(probs_raw) > len(y_true):
        probs_raw = probs_raw[:len(y_true)]
    probs_cal = probs_ens
    auc_diff = bootstrap_auc_diff_test(y_true, probs_cal, probs_raw)
    results.append({
        "test": "Bootstrap AUC diff (calibrated vs raw)",
        "statistic": auc_diff.get("z_statistic", float("nan")),
        "p_value": auc_diff["p_value"],
        "metric_a": auc_diff.get("auc_a", float("nan")),
        "metric_b": auc_diff.get("auc_b", float("nan")),
        "note": f"method={auc_diff.get('method')}, AUC diff: {auc_diff.get('auc_diff', 0):.5f}",
    })

    ai_acc_ci = bootstrap_metric_ci(y_true, y_ai, accuracy_score)
    hitl_acc_ci = bootstrap_metric_ci(y_true, y_hitl, accuracy_score)
    ai_f1_ci = bootstrap_metric_ci(y_true, y_ai, lambda yt, yp: f1_score(yt, yp, average="macro"))
    hitl_f1_ci = bootstrap_metric_ci(y_true, y_hitl, lambda yt, yp: f1_score(yt, yp, average="macro"))
    results.append({
        "test": "AI-only accuracy CI",
        "statistic": ai_acc_ci["mean"],
        "p_value": float("nan"),
        "metric_a": ai_acc,
        "metric_b": float("nan"),
        "note": f"95% CI: [{ai_acc_ci['ci_lower_95']:.4f}, {ai_acc_ci['ci_upper_95']:.4f}]",
    })
    results.append({
        "test": "HITL accuracy CI",
        "statistic": hitl_acc_ci["mean"],
        "p_value": float("nan"),
        "metric_a": hitl_acc,
        "metric_b": float("nan"),
        "note": f"95% CI: [{hitl_acc_ci['ci_lower_95']:.4f}, {hitl_acc_ci['ci_upper_95']:.4f}]",
    })
    results.append({
        "test": "AI-only macro F1 CI",
        "statistic": ai_f1_ci["mean"],
        "p_value": float("nan"),
        "metric_a": ai_f1,
        "metric_b": float("nan"),
        "note": f"95% CI: [{ai_f1_ci['ci_lower_95']:.4f}, {ai_f1_ci['ci_upper_95']:.4f}]",
    })
    results.append({
        "test": "HITL macro F1 CI",
        "statistic": hitl_f1_ci["mean"],
        "p_value": float("nan"),
        "metric_a": hitl_f1,
        "metric_b": float("nan"),
        "note": f"95% CI: [{hitl_f1_ci['ci_lower_95']:.4f}, {hitl_f1_ci['ci_upper_95']:.4f}]",
    })

    df_out = pd.DataFrame(results)
    out_path = os.path.join(TABLES_DIR, "statistical_tests.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved statistical tests to: {out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_statistical_tests()
