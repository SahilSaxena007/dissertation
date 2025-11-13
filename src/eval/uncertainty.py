"""
Component 8️⃣: Uncertainty Signals (entropy, margin, confidence).
---------------------------------------------------------------
Quantifies model prediction uncertainty for:
  • HITL triage (Step 2 trigger thresholds)
  • Meta-model training (Step 2 escalation predictor)
  • Dataset difficulty / error analysis

Author: Soumil Saxena (University of Manchester)
"""

import numpy as np
import pandas as pd


# ╭──────────────────────────────────────────────╮
# │  Core Computation: Entropy, Margin, Confidence │
# ╰──────────────────────────────────────────────╯
def compute_uncertainty_signals(y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute entropy, margin (top-2 prob gap), and confidence per sample.

    Parameters
    ----------
    y_prob : array-like, shape (n_samples, n_classes)
        Model-predicted class probabilities (each row sums to ~1).

    Returns
    -------
    pd.DataFrame
        Columns:
        • entropy : Shannon entropy  [0, log(C)]
                    (0 = certain, log(C) = fully uncertain)
        • margin  : Difference between top-2 predicted probabilities [0, 1]
                    (0 = tie, 1 = very confident)
        • max_prob : Probability of the predicted class
        • confidence_gap : Alias for margin (for convenience)
    """
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0)  # avoid log(0)
    n_samples, n_classes = y_prob.shape

    # ── 1️⃣ ENTROPY ───────────────────────────────
    # Vectorized Shannon entropy: -Σ p * log(p)
    entropy = -np.sum(y_prob * np.log(y_prob), axis=1)
    entropy /= np.log(n_classes)  # normalize to [0,1] across class count

    # ── 2️⃣ MAX PROBABILITY (Confidence) ─────────
    max_prob = np.max(y_prob, axis=1)

    # ── 3️⃣ MARGIN (Top-2 Probability Gap) ───────
    # Efficient top-2 extraction without sorting entire row
    top2 = np.partition(y_prob, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    confidence_gap = margin.copy()

    # ── Construct DataFrame ──────────────────────
    df = pd.DataFrame({
        "entropy": entropy,
        "margin": margin,
        "max_prob": max_prob,
        "confidence_gap": confidence_gap,
    })

    return df


# ╭──────────────────────────────────────────────╮
# │  Classification into Uncertainty Levels       │
# ╰──────────────────────────────────────────────╯
def classify_uncertainty_level(
    uncertainty_df: pd.DataFrame,
    entropy_thresh: float | None = None,
    margin_thresh: float | None = None,
) -> pd.DataFrame:
    """
    Assign qualitative uncertainty levels (LOW / MEDIUM / HIGH).

    Logic:
      HIGH   → high entropy  AND  low margin
      MEDIUM → one of them (entropy or margin) indicates uncertainty
      LOW    → confident (low entropy, high margin)

    Parameters
    ----------
    uncertainty_df : pd.DataFrame
        Output of compute_uncertainty_signals().
    entropy_thresh : float, optional
        Threshold above which entropy is "high" (default: 75th percentile).
    margin_thresh : float, optional
        Threshold below which margin is "low" (default: 25th percentile).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with additional column 'uncertainty_level'.
    """
    df = uncertainty_df.copy()

    if entropy_thresh is None:
        entropy_thresh = df["entropy"].quantile(0.75)
    if margin_thresh is None:
        margin_thresh = df["margin"].quantile(0.25)

    conditions = [
        (df["entropy"] > entropy_thresh) & (df["margin"] < margin_thresh),
        (df["entropy"] > entropy_thresh) | (df["margin"] < margin_thresh),
    ]
    choices = ["HIGH", "MEDIUM"]
    df["uncertainty_level"] = np.select(conditions, choices, default="LOW")

    return df


# ╭──────────────────────────────────────────────╮
# │  Summary Statistics                          │
# ╰──────────────────────────────────────────────╯
def uncertainty_summary_stats(uncertainty_df: pd.DataFrame) -> dict:
    """
    Summarize uncertainty signals with descriptive statistics.

    Parameters
    ----------
    uncertainty_df : pd.DataFrame
        Output from compute_uncertainty_signals().

    Returns
    -------
    dict
        {signal: {mean, std, min, max, p25, p50, p75}}
    """
    summary = {}
    for col in ["entropy", "margin", "max_prob"]:
        s = uncertainty_df[col]
        summary[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
        }
    return summary
