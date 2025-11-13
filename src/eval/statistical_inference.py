"""
Component 6️⃣: Bootstrap confidence intervals for metrics.
"""

def bootstrap_confidence_intervals(
    metric_func,
    y_true,
    y_pred_or_proba,
    n_bootstrap=1000,
    random_state=42,
    ci=95
):
    """
    Resample and compute confidence intervals for a metric.
    
    Parameters:
    - metric_func: Function that takes (y_true, y_pred_or_proba) and returns a scalar
    - y_true: True labels
    - y_pred_or_proba: Predicted labels or probabilities
    - n_bootstrap: Number of bootstrap resamples
    - random_state: Seed
    - ci: Confidence level (e.g., 95)
    
    Returns:
    - Dict with keys: 'mean', 'ci_low', 'ci_high', 'std'
    """
    # TODO: Implement
    pass