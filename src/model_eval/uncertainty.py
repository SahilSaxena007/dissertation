# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Uncertainty signals (entropy, margin, confidence) used for HITL triage
and meta-model training.
"""

import numpy as np
import pandas as pd


# Core Computation: Entropy, Margin, Confidence
def compute_uncertainty_signals(y_prob: np.ndarray) -> pd.DataFrame:
    """Returns entropy, margin, max_prob, and confidence_gap per sample."""
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0)  # avoid log(0)
    n_samples, n_classes = y_prob.shape

    # Entropy: Vectorized Shannon entropy -sum(p * log(p))
    entropy = -np.sum(y_prob * np.log(y_prob), axis=1)
    entropy /= np.log(n_classes)  # normalize to [0,1] across class count

    # Max probability (confidence)
    max_prob = np.max(y_prob, axis=1)

    # Margin: top-2 probability gap
    # Efficient top-2 extraction without sorting entire row
    top2 = np.partition(y_prob, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    confidence_gap = margin.copy()

    # Construct DataFrame
    df = pd.DataFrame({
        "entropy": entropy,
        "margin": margin,
        "max_prob": max_prob,
        "confidence_gap": confidence_gap,
    })

    return df


# Classification into Uncertainty Levels
def classify_uncertainty_level(
    uncertainty_df: pd.DataFrame,
    entropy_thresh: float | None = None,
    margin_thresh: float | None = None,
) -> pd.DataFrame:
    """
    Assigns LOW/MEDIUM/HIGH uncertainty level per sample.
    Defaults to 75th/25th percentile thresholds if not provided.
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


# Summary Statistics
def uncertainty_summary_stats(uncertainty_df: pd.DataFrame) -> dict:
    """Returns descriptive stats (mean, std, percentiles) for each uncertainty signal."""
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
