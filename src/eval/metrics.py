"""
Component 1️⃣ 2️⃣: Overall and per-class performance metrics.
"""

def compute_classification_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute overall metrics: accuracy, precision, recall, F1, AUC, Brier score, log-loss.
    
    Parameters:
    - y_true: True labels (array-like)
    - y_pred: Predicted labels (array-like)
    - y_prob: Predicted probabilities (n_samples x n_classes)
    - class_names: List of class names (e.g., ["SCD", "MCI", "AD"])
    
    Returns:
    - metrics: Dict with overall metrics
    - per_class_df: DataFrame with per-class metrics
    """
    # TODO: Implement
    pass


def per_class_metrics_table(y_true, y_pred, y_prob, class_names):
    """
    Compute per-class precision, recall, F1, AUC.
    
    Returns:
    - DataFrame with one row per class
    """
    # TODO: Implement
    pass