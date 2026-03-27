"""
Component 7️⃣: Export per-sample error records for analysis.
Creates detailed CSV with predictions, uncertainties, and error flags.
Used to train meta-models (Step 2: learn-to-defer) and analyze error patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from .uncertainty import compute_uncertainty_signals


def export_error_taxonomy(
    y_true,
    y_pred,
    y_prob,
    class_names,
    model_name,
    save_path,
    sample_ids=None,
    metadata_df=None,
):
    """
    Create a detailed CSV with one row per test sample.
    """

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    y_prob = np.array(y_prob)

    n_samples = len(y_true)
    n_classes = len(class_names)

    # 1️⃣ UNCERTAINTY SIGNALS
    uncertainty_df = compute_uncertainty_signals(y_prob)
    entropy = np.array(uncertainty_df["entropy"]).reshape(-1)
    margin = np.array(uncertainty_df["margin"]).reshape(-1)
    max_prob = np.array(uncertainty_df["max_prob"]).reshape(-1)

    # 2️⃣ SAMPLE IDS
    if sample_ids is None:
        sample_ids = np.arange(n_samples)

    # 3️⃣ BUILD CLEAN COLUMN-WISE DATA
    error_data = {}

    error_data["sample_id"] = np.array(sample_ids).reshape(-1)
    error_data["sample_index"] = np.arange(n_samples)

    # Labels
    error_data["true_label"] = [class_names[int(y)] for y in y_true]
    error_data["predicted_label"] = [class_names[int(y)] for y in y_pred]
    error_data["true_label_int"] = y_true
    error_data["predicted_label_int"] = y_pred

    # Class probabilities – guaranteed 1D arrays
    for i, cname in enumerate(class_names):
        error_data[f"prob_{cname}"] = y_prob[:, i].reshape(-1)

    # Uncertainty – guaranteed 1D arrays
    error_data["entropy"] = entropy
    error_data["margin"] = margin
    error_data["max_prob"] = max_prob

    # Error flags
    is_correct = (y_true == y_pred).astype(int)
    error_data["is_correct"] = is_correct
    error_data["is_error"] = 1 - is_correct

    # Model name and timestamp – REPEAT PER SAMPLE (previous bug fixed)
    error_data["model_name"] = [model_name] * n_samples
    error_data["timestamp"] = [datetime.now().isoformat()] * n_samples

    # Create DataFrame
    error_df = pd.DataFrame(error_data)

    # Optional metadata
    if metadata_df is not None:
        if len(metadata_df) != n_samples:
            raise ValueError(
                f"metadata_df length {len(metadata_df)} must match {n_samples}"
            )
        error_df = pd.concat([error_df, metadata_df.reset_index(drop=True)], axis=1)

    # Save
    error_df.to_csv(save_path, index=False)
    print(f"✅ Error taxonomy exported to {save_path}")
    print(f"   Rows: {len(error_df)}, Columns: {len(error_df.columns)}")

    return error_df



def error_taxonomy_summary(error_df, class_names):
    """
    Compute summary statistics from error taxonomy.
    
    Parameters
    ----------
    error_df : pd.DataFrame
        Output from export_error_taxonomy().
    class_names : list of str
        Class names.
    
    Returns
    -------
    summary : dict
        Contains error rates, uncertainty stats by correctness, confusion matrix, etc.
    """
    
    n_correct = error_df['is_correct'].sum()
    n_error = error_df['is_error'].sum()
    n_total = len(error_df)
    error_rate = n_error / n_total
    
    summary = {
        'total_samples': n_total,
        'correct_predictions': n_correct,
        'error_predictions': n_error,
        'accuracy': n_correct / n_total,
        'error_rate': error_rate,
    }
    
    # Uncertainty stats for correct vs. error predictions
    correct_entropy = error_df[error_df['is_correct'] == 1]['entropy'].mean()
    error_entropy = error_df[error_df['is_error'] == 1]['entropy'].mean()
    
    correct_margin = error_df[error_df['is_correct'] == 1]['margin'].mean()
    error_margin = error_df[error_df['is_error'] == 1]['margin'].mean()
    
    correct_max_prob = error_df[error_df['is_correct'] == 1]['max_prob'].mean()
    error_max_prob = error_df[error_df['is_error'] == 1]['max_prob'].mean()
    
    summary['uncertainty_stats'] = {
        'correct': {
            'entropy': correct_entropy,
            'margin': correct_margin,
            'max_prob': correct_max_prob,
        },
        'error': {
            'entropy': error_entropy,
            'margin': error_margin,
            'max_prob': error_max_prob,
        },
    }
    
    # Confusion matrix
    confusion_dict = {}
    for true_label in class_names:
        for pred_label in class_names:
            count = len(
                error_df[
                    (error_df['true_label'] == true_label) &
                    (error_df['predicted_label'] == pred_label)
                ]
            )
            confusion_dict[f'{true_label}_→_{pred_label}'] = count
    
    summary['confusion_matrix'] = confusion_dict
    
    return summary


def error_cases_by_uncertainty(error_df, n_top=10):
    """
    Extract most uncertain error cases (for manual review / HITL).
    
    Parameters
    ----------
    error_df : pd.DataFrame
        Output from export_error_taxonomy().
    n_top : int, default=10
        Number of top uncertain errors to return.
    
    Returns
    -------
    top_uncertain : pd.DataFrame
        Top n_top error cases ranked by entropy (highest uncertainty first).
    """
    
    # Filter only errors
    errors_only = error_df[error_df['is_error'] == 1].copy()
    
    # Sort by entropy (descending)
    top_uncertain = errors_only.nlargest(n_top, 'entropy')
    
    return top_uncertain


def export_meta_model_training_data(error_dfs_list, class_names, save_path):
    """
    Combine error taxonomies from multiple models into a single training dataset
    for meta-model (learn-to-defer).
    
    Parameters
    ----------
    error_dfs_list : list of pd.DataFrame
        Error taxonomy DataFrames from multiple models.
    class_names : list of str
        Class names.
    save_path : str
        Path to save combined training data.
    
    Returns
    -------
    meta_training_df : pd.DataFrame
        Combined dataset ready for meta-model training.
    """
    
    # Concatenate all error dataframes
    meta_training_df = pd.concat(error_dfs_list, axis=0, ignore_index=True)
    
    # Compute features for meta-model
    # Meta-model input: uncertainty signals + model predictions
    # Meta-model target: is_correct (0/1)
    
    meta_training_df.to_csv(save_path, index=False)
    print(f"✅ Meta-model training data exported to {save_path}")
    print(f"   Total samples: {len(meta_training_df)}")
    print(f"   Unique models: {meta_training_df['model_name'].nunique()}")
    
    return meta_training_df