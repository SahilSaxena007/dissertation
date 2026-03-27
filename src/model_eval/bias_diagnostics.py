"""
Component 1️⃣0️⃣: Bias diagnostics and fairness analysis.
Includes demographic parity, group fairness metrics, and disparate impact detection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score


def demographic_parity(error_df, group_col, positive_label=None):
    """
    Compute demographic parity: proportion of positive predictions per group.
    
    Parameters
    ----------
    error_df : pd.DataFrame
        Error taxonomy dataframe (must include predicted_label column).
    group_col : str
        Column name for demographic group (e.g., 'sex', 'race', 'scanner').
    positive_label : str or int, optional
        Label considered "positive" (e.g., 'AD' or 2). If None, uses predicted_label == predicted_label.
    
    Returns
    -------
    parity_df : pd.DataFrame
        Columns: [group, positive_rate, n_samples]
    """
    if positive_label is None:
        # Use most severe class (last in class_names)
        positive_label = error_df['predicted_label'].unique()[-1]
    
    groups = error_df[group_col].unique()
    rows = []
    for group in groups:
        group_mask = error_df[group_col] == group
        n_group = group_mask.sum()
        n_positive = (error_df[group_mask]['predicted_label'] == positive_label).sum()
        positive_rate = n_positive / n_group if n_group > 0 else np.nan
        rows.append({'group': group, 'positive_rate': positive_rate, 'n_samples': n_group})
    parity_df = pd.DataFrame(rows)
    return parity_df


def fairness_metrics_by_group(error_df, group_col, class_names):
    """
    Compute accuracy, precision, recall per group.
    
    Parameters
    ----------
    error_df : pd.DataFrame
        Error taxonomy dataframe.
    group_col : str
        Demographic column (e.g., 'sex', 'race').
    class_names : list of str
        Class names.
    
    Returns
    -------
    metrics_df : pd.DataFrame
        Columns: [group, accuracy, macro_precision, macro_recall]
    """
    groups = error_df[group_col].unique()
    rows = []
    for group in groups:
        group_mask = error_df[group_col] == group
        y_true = error_df[group_mask]['true_label_int']
        y_pred = error_df[group_mask]['predicted_label_int']
        if len(y_true) == 0:
            continue
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        rows.append({
            'group': group,
            'accuracy': acc,
            'macro_precision': prec,
            'macro_recall': rec,
            'n_samples': len(y_true),
        })
    metrics_df = pd.DataFrame(rows)
    return metrics_df


def disparate_impact(error_df, group_col, positive_label=None):
    """
    Compute disparate impact ratio between groups (80% rule).
    
    Parameters
    ----------
    error_df : pd.DataFrame
        Error taxonomy dataframe.
    group_col : str
        Demographic column.
    positive_label : str or int, optional
        Label considered "positive". If None, uses last class.
    
    Returns
    -------
    impact_df : pd.DataFrame
        Columns: [group, positive_rate]
        Also returns disparate impact ratio (min/max).
    """
    parity_df = demographic_parity(error_df, group_col, positive_label)
    rates = parity_df['positive_rate'].values
    min_rate = np.nanmin(rates)
    max_rate = np.nanmax(rates)
    disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else np.nan
    parity_df['disparate_impact_ratio'] = disparate_impact_ratio
    return parity_df, disparate_impact_ratio