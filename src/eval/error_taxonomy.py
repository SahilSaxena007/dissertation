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
    
    Columns include:
    - Sample identifiers (sample_id, index)
    - True and predicted labels
    - All class probabilities
    - Uncertainty signals (entropy, margin, confidence)
    - Correctness flag
    - Model name & timestamp
    - Optional metadata (age, sex, scanner, etc.)
    
    Parameters
    ----------
    y_true : array-like
        True class labels (numeric 0..C-1).
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities.
    class_names : list of str
        Class names (e.g., ["SCD", "MCI", "AD"]).
    model_name : str
        Name of the model (e.g., "CatBoost", "RandomForest", "NeuralNetwork").
    save_path : str
        Path to save CSV (e.g., "../reports/tables/error_taxonomy_catboost.csv").
    sample_ids : array-like, optional
        Sample identifiers (e.g., RID, patient ID). Default: 0-indexed integers.
    metadata_df : pd.DataFrame, optional
        Additional metadata columns (age, sex, scanner, etc.) to include.
        Should have same length as y_true.
    
    Returns
    -------
    error_df : pd.DataFrame
        The exported error taxonomy DataFrame.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    n_samples = len(y_true)
    n_classes = len(class_names)
    
    # ─────────────────────────────────────────
    # 1️⃣ COMPUTE UNCERTAINTY SIGNALS
    # ─────────────────────────────────────────
    uncertainty_df = compute_uncertainty_signals(y_prob)
    
    # ─────────────────────────────────────────
    # 2️⃣ BUILD ERROR TAXONOMY DATAFRAME
    # ─────────────────────────────────────────
    
    # Sample identifiers
    if sample_ids is None:
        sample_ids = np.arange(n_samples)
    
    error_data = {
        'sample_id': sample_ids,
        'sample_index': np.arange(n_samples),
    }
    
    # True and predicted labels
    error_data['true_label'] = [class_names[int(y)] for y in y_true]
    error_data['predicted_label'] = [class_names[int(y)] for y in y_pred]
    error_data['true_label_int'] = y_true
    error_data['predicted_label_int'] = y_pred
    
    # Class probabilities
    for i, class_name in enumerate(class_names):
        error_data[f'prob_{class_name}'] = y_prob[:, i]
    
    # Uncertainty signals
    error_data['entropy'] = uncertainty_df['entropy'].values
    error_data['margin'] = uncertainty_df['margin'].values
    error_data['max_prob'] = uncertainty_df['max_prob'].values
    
    # Correctness flag
    error_data['is_correct'] = (y_true == y_pred).astype(int)
    error_data['is_error'] = (~(y_true == y_pred)).astype(int)
    
    # Model and timestamp
    error_data['model_name'] = model_name
    error_data['timestamp'] = datetime.now().isoformat()
    
    # Create DataFrame
    error_df = pd.DataFrame(error_data)
    
    # ─────────────────────────────────────────
    # 3️⃣ ADD OPTIONAL METADATA
    # ─────────────────────────────────────────
    if metadata_df is not None:
        # Ensure same length
        if len(metadata_df) != n_samples:
            raise ValueError(
                f"metadata_df length ({len(metadata_df)}) must match "
                f"y_true length ({n_samples})"
            )
        
        # Concatenate metadata columns
        error_df = pd.concat([error_df, metadata_df.reset_index(drop=True)], axis=1)
    
    # ─────────────────────────────────────────
    # 4️⃣ SAVE TO CSV
    # ─────────────────────────────────────────
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