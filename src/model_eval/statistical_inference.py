# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Component 6: Bootstrap confidence intervals for metrics.
Adds statistical rigor by quantifying robustness of metrics via resampling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    y_prob,
    class_names,
    n_bootstrap=1000,
    ci=95,
    random_state=42,
):
    """
    Percentile-based bootstrap CIs for accuracy, F1, AUC and per-class AUC.
    Returns (ci_results dict, bootstrap_samples dict).
    """
    
    np.random.seed(random_state)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_samples = len(y_true)
    n_classes = len(class_names)
    
    # Binarize for per-class AUC
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Storage for bootstrap samples
    bootstrap_samples = {
        'accuracy': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
        'weighted_f1': [],
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        bootstrap_samples[f'{class_name}_precision'] = []
        bootstrap_samples[f'{class_name}_recall'] = []
        bootstrap_samples[f'{class_name}_f1'] = []
        bootstrap_samples[f'{class_name}_auc'] = []
    
    # ─────────────────────────────────────────
    # BOOTSTRAP RESAMPLING
    # ─────────────────────────────────────────
    for b in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]
        y_prob_b = y_prob[indices]
        y_true_bin_b = y_true_bin[indices]
        
        # Overall metrics
        bootstrap_samples['accuracy'].append(
            accuracy_score(y_true_b, y_pred_b)
        )
        bootstrap_samples['macro_precision'].append(
            precision_score(y_true_b, y_pred_b, average='macro', zero_division=0)
        )
        bootstrap_samples['macro_recall'].append(
            recall_score(y_true_b, y_pred_b, average='macro', zero_division=0)
        )
        bootstrap_samples['macro_f1'].append(
            f1_score(y_true_b, y_pred_b, average='macro', zero_division=0)
        )
        bootstrap_samples['weighted_f1'].append(
            f1_score(y_true_b, y_pred_b, average='weighted', zero_division=0)
        )
        
        # Per-class metrics
        precision_per_class = precision_score(
            y_true_b, y_pred_b, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true_b, y_pred_b, average=None, zero_division=0
        )
        f1_per_class = f1_score(y_true_b, y_pred_b, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            bootstrap_samples[f'{class_name}_precision'].append(precision_per_class[i])
            bootstrap_samples[f'{class_name}_recall'].append(recall_per_class[i])
            bootstrap_samples[f'{class_name}_f1'].append(f1_per_class[i])
            
            # Per-class AUC
            try:
                auc_i = roc_auc_score(y_true_bin_b[:, i], y_prob_b[:, i])
            except Exception:
                auc_i = np.nan
            
            bootstrap_samples[f'{class_name}_auc'].append(auc_i)
    
    # ─────────────────────────────────────────
    # COMPUTE CIs FROM BOOTSTRAP SAMPLES
    # ─────────────────────────────────────────
    alpha = (100 - ci) / 2  # e.g., 2.5 for 95% CI
    ci_results = {}
    
    for metric_name, samples in bootstrap_samples.items():
        samples_arr = np.array(samples)
        
        # Remove NaN values if any
        samples_arr = samples_arr[~np.isnan(samples_arr)]
        
        if len(samples_arr) == 0:
            ci_results[metric_name] = {
                'mean': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'std': np.nan,
            }
        else:
            ci_results[metric_name] = {
                'mean': np.mean(samples_arr),
                'ci_low': np.percentile(samples_arr, alpha),
                'ci_high': np.percentile(samples_arr, 100 - alpha),
                'std': np.std(samples_arr),
            }
    
    return ci_results, bootstrap_samples


def bootstrap_ci_table(ci_results, class_names):
    """
    Create a formatted DataFrame from bootstrap CI results.
    
    Parameters
    ----------
    ci_results : dict
        Output from bootstrap_confidence_intervals().
    class_names : list of str
        Class names.
    
    Returns
    -------
    ci_df : pd.DataFrame
        Columns: ['Metric', 'Mean', 'CI Low', 'CI High', 'Std']
    """
    
    rows = []
    
    # Overall metrics
    for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']:
        if metric in ci_results:
            result = ci_results[metric]
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Mean': result['mean'],
                'CI Low': result['ci_low'],
                'CI High': result['ci_high'],
                'Std': result['std'],
            })
    
    # Per-class metrics
    for class_name in class_names:
        for metric_type in ['precision', 'recall', 'f1', 'auc']:
            metric_key = f'{class_name}_{metric_type}'
            if metric_key in ci_results:
                result = ci_results[metric_key]
                rows.append({
                    'Metric': f'{class_name} {metric_type.upper()}',
                    'Mean': result['mean'],
                    'CI Low': result['ci_low'],
                    'CI High': result['ci_high'],
                    'Std': result['std'],
                })
    
    ci_df = pd.DataFrame(rows)
    return ci_df


def ci_width_summary(ci_results):
    """
    Compute CI width statistics (useful for assessing metric stability).
    
    Parameters
    ----------
    ci_results : dict
        Output from bootstrap_confidence_intervals().
    
    Returns
    -------
    summary : dict
        Contains 'mean_ci_width', 'median_ci_width', 'max_ci_width'
    """
    
    ci_widths = []
    for metric_name, result in ci_results.items():
        if not np.isnan(result['ci_low']) and not np.isnan(result['ci_high']):
            width = result['ci_high'] - result['ci_low']
            ci_widths.append(width)
    
    if len(ci_widths) == 0:
        return {
            'mean_ci_width': np.nan,
            'median_ci_width': np.nan,
            'max_ci_width': np.nan,
            'min_ci_width': np.nan,
        }
    
    ci_widths_arr = np.array(ci_widths)
    
    summary = {
        'mean_ci_width': np.mean(ci_widths_arr),
        'median_ci_width': np.median(ci_widths_arr),
        'max_ci_width': np.max(ci_widths_arr),
        'min_ci_width': np.min(ci_widths_arr),
    }
    
    return summary