"""
Low-level rule functions for Step 2.

Each function:
  - takes simple inputs (probabilities, predictions, feature row)
  - returns numeric metrics + boolean flags
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from .config import EscalationConfig
except ImportError:
    from config import EscalationConfig


# ---------------------------------------------------------------------------
# 1. Uncertainty metrics (max prob, margin, entropy)
# ---------------------------------------------------------------------------


def compute_uncertainty_metrics(
    probs: np.ndarray, config: EscalationConfig
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    probs : (n_classes,) array of ensemble probabilities.
    config : EscalationConfig

    Returns
    -------
    dict with:
        max_prob, margin, entropy,
        low_confidence, small_margin, high_entropy
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"Expected 1-D probs, got shape {probs.shape}")

    max_prob = float(probs.max())
    sorted_probs = np.sort(probs)[::-1]
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))

    th = config.uncertainty
    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "low_confidence": max_prob < th.low_confidence,
        "small_margin": margin < th.small_margin,
        "high_entropy": entropy > th.high_entropy,
    }


# ---------------------------------------------------------------------------
# 2. Model disagreement
# ---------------------------------------------------------------------------


def compute_model_disagreement(preds: Dict[str, int]) -> Dict[str, Any]:
    """
    Parameters
    ----------
    preds : mapping model_name -> predicted_label_index

    Returns
    -------
    dict with:
        unique_labels, disagreement, disagreement_score
    """
    labels = list(preds.values())
    unique = set(labels)
    disagreement = len(unique) > 1

    values, counts = np.unique(labels, return_counts=True)
    majority_fraction = counts.max() / len(labels)
    disagreement_score = float(1.0 - majority_fraction)

    return {
        "unique_labels": unique,
        "disagreement": disagreement,
        "disagreement_score": disagreement_score,
    }


# ---------------------------------------------------------------------------
# 3. Missing-data check
# ---------------------------------------------------------------------------


def compute_missingness_flags(
    x_row: np.ndarray,
    config: EscalationConfig,
    pre_imputation_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Missingness logic using pre-imputation NaN mask when available.

    Parameters
    ----------
    x_row : (n_features,) array (post-imputation, used as fallback)
    config : EscalationConfig
    pre_imputation_mask : optional (n_features,) boolean array from before imputation
    """
    if pre_imputation_mask is not None:
        mask = np.asarray(pre_imputation_mask, dtype=bool)
    else:
        x_row = np.asarray(x_row, dtype=float)
        mask = np.isnan(x_row)

    missing_fraction = float(mask.mean())
    has_missing = bool(mask.any())
    critical_missing = missing_fraction >= config.missing.critical_missing_fraction

    return {
        "missing_fraction": missing_fraction,
        "has_missing": has_missing,
        "critical_missing": critical_missing,
    }


# ---------------------------------------------------------------------------
# 4. Multimodal mismatch
# ---------------------------------------------------------------------------


def _parse_group_spec(group_spec: Any) -> Tuple[Iterable[int], float]:
    """
    Parse group definition.

    Supported formats:
      - [idx0, idx1, ...]
      - {"indices": [...], "direction": 1 or -1}

    direction:
      +1 means higher values indicate higher severity.
      -1 means lower values indicate higher severity.
    """
    if isinstance(group_spec, dict):
        indices = group_spec.get("indices", [])
        direction = float(group_spec.get("direction", 1.0))
        direction = 1.0 if direction >= 0 else -1.0
        return indices, direction
    return group_spec, 1.0


def compute_multimodal_mismatch(
    x_row: np.ndarray, config: EscalationConfig
) -> Dict[str, Any]:
    """
    Detect conflict between modality groups (e.g., biomarker vs cognitive).

    Group scores are robust-z normalized and direction-aligned so the scores
    can be compared on a common severity scale.
    """
    groups = config.multimodal_feature_groups
    if not groups or len(groups) < 2:
        return {
            "mismatch": False,
            "group_scores": {},
            "mismatch_score": 0.0,
            "mismatch_threshold": float("nan"),
        }

    x_row = np.asarray(x_row, dtype=float)
    finite_vals = x_row[np.isfinite(x_row)]
    if finite_vals.size == 0:
        return {
            "mismatch": False,
            "group_scores": {},
            "mismatch_score": 0.0,
            "mismatch_threshold": float("nan"),
        }

    row_median = float(np.median(finite_vals))
    row_mad = float(np.median(np.abs(finite_vals - row_median)))
    robust_scale = max(1.4826 * row_mad, 1e-6)

    group_scores: Dict[str, float] = {}
    for name, group_spec in groups.items():
        idxs, direction = _parse_group_spec(group_spec)
        valid = [int(i) for i in idxs if 0 <= int(i) < len(x_row)]
        if not valid:
            continue

        vals = x_row[valid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        aligned_mean = float(np.mean(vals) * direction)
        robust_z = float((aligned_mean - row_median) / robust_scale)
        group_scores[name] = robust_z

    if len(group_scores) < 2:
        return {
            "mismatch": False,
            "group_scores": group_scores,
            "mismatch_score": 0.0,
            "mismatch_threshold": float("nan"),
        }

    scores = np.array(list(group_scores.values()), dtype=float)
    diff_matrix = np.abs(scores[:, None] - scores[None, :])
    mismatch_score = float(np.nanmax(diff_matrix))

    score_std = float(np.nanstd(scores))
    mismatch_threshold = max(1.5, 1.0 + score_std)
    mismatch = bool(mismatch_score >= mismatch_threshold)

    return {
        "mismatch": mismatch,
        "group_scores": group_scores,
        "mismatch_score": mismatch_score,
        "mismatch_threshold": mismatch_threshold,
    }


# ---------------------------------------------------------------------------
# 5. SHAP instability
# ---------------------------------------------------------------------------


def _to_feature_vector(shap_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Common SHAP layouts:
        #   (n_classes, n_features) or (n_features, n_classes)
        if arr.shape[0] <= 5 and arr.shape[1] > arr.shape[0]:
            return np.nanmean(arr, axis=0)
        if arr.shape[1] <= 5 and arr.shape[0] > arr.shape[1]:
            return np.nanmean(arr, axis=1)
        return np.nanmean(arr, axis=0)
    # For higher-rank tensors, collapse all but last axis.
    return np.nanmean(arr.reshape(-1, arr.shape[-1]), axis=0)


def compute_shap_instability(
    shap_by_model: Optional[Dict[str, np.ndarray]] = None,
    variance_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute SHAP instability from variance across model explanations.

    Parameters
    ----------
    shap_by_model : dict or None
        Mapping model_name -> SHAP attribution vector/tensor for one sample.
    variance_threshold : float
        Instability flag threshold for mean per-feature variance.
    """
    if not shap_by_model or len(shap_by_model) < 2:
        return {"instability": False, "variance": float("nan"), "available": False}

    vectors = []
    used_models = []
    min_len: Optional[int] = None

    for model_name, shap_values in shap_by_model.items():
        if shap_values is None:
            continue
        vec = _to_feature_vector(shap_values)
        if vec.size == 0 or not np.isfinite(vec).any():
            continue

        min_len = vec.size if min_len is None else min(min_len, vec.size)
        vectors.append(vec)
        used_models.append(model_name)

    if len(vectors) < 2 or min_len is None or min_len == 0:
        return {"instability": False, "variance": float("nan"), "available": False}

    aligned = np.vstack([v[:min_len] for v in vectors])
    per_feature_var = np.nanvar(aligned, axis=0)
    variance = float(np.nanmean(per_feature_var))
    instability = bool(np.isfinite(variance) and variance > variance_threshold)

    return {
        "instability": instability,
        "variance": variance,
        "available": True,
        "used_models": used_models,
        "n_features": int(min_len),
    }
