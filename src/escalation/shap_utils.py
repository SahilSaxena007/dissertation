"""
SHAP utilities for batch computation of feature attributions.

Uses TreeExplainer for CatBoost and RandomForest (fast).
Skips NN (KernelExplainer too slow; 2 tree models sufficient for instability).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import shap
except ImportError:
    shap = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def compute_batch_shap(
    models: Dict[str, Any],
    X: np.ndarray,
    source: str = "oof",
) -> Dict[str, np.ndarray]:
    """
    Compute SHAP values for CatBoost and RF models.

    Parameters
    ----------
    models : dict with keys "catboost" and "rf"
    X : (N, n_features) scaled feature matrix
    source : "oof" or "holdout" (used for cache filenames)

    Returns
    -------
    dict mapping model name -> SHAP values array (N, n_features, n_classes) or (N, n_features)
    """
    if shap is None:
        print("WARNING: shap package not installed. Skipping SHAP computation.")
        return {}

    result = {}

    for model_name in ["catboost", "rf", "xgb"]:
        if model_name not in models:
            continue

        cache_path = os.path.join(ARTIFACTS_DIR, f"shap_{source}_{model_name}.npy")
        if os.path.exists(cache_path):
            print(f"   Loading cached SHAP values: {cache_path}")
            result[model_name] = np.load(cache_path, allow_pickle=True)
            continue

        model = models[model_name]
        print(f"   Computing SHAP values for {model_name}...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # shap_values shape varies:
        # - For multiclass: list of (N, F) arrays or (N, F, C) array
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)  # -> (N, F, C)

        shap_values = np.asarray(shap_values, dtype=float)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        np.save(cache_path, shap_values)
        print(f"   Cached SHAP values to: {cache_path}")
        result[model_name] = shap_values

    return result


def load_or_compute_shap(
    models: Dict[str, Any],
    X: np.ndarray,
    source: str = "oof",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load cached SHAP or compute fresh. Returns None if shap is unavailable.
    """
    if shap is None:
        return None

    result = compute_batch_shap(models, X, source=source)
    return result if result else None
