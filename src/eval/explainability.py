"""
Component 9️⃣: SHAP-based explainability analysis.
Provides global feature importance, sample-level explanations, and dependency plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def compute_shap_values(model, X, class_names, model_type='tree', max_samples=None):
    """
    Compute SHAP values for model explanations.
    
    Parameters
    ----------
    model : sklearn model or similar
        Trained model with predict/predict_proba methods.
    X : array-like, shape (n_samples, n_features)
        Feature matrix for explanation.
    class_names : list of str
        Class names (e.g., ["SCD", "MCI", "AD"]).
    model_type : str, default='tree'
        Type of model: 'tree' (CatBoost, XGBoost, RandomForest) or 'kernel' (any model).
    max_samples : int, optional
        Max samples to use for SHAP computation (for speed). Default: use all.
    
    Returns
    -------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : array-like
        SHAP values (n_samples, n_features) for binary/regression
        or (n_samples, n_features, n_classes) for multiclass.
    X_sample : array-like
        Feature matrix used (may be subsampled).
    """
    
    X = np.array(X)
    n_samples, n_features = X.shape
    
    print(f"Computing SHAP values for {n_samples} samples, {n_features} features...")
    print(f"  Model type: {model_type}")
    
    # Subsample if needed (for speed)
    if max_samples is not None and n_samples > max_samples:
        indices = np.random.choice(n_samples, size=max_samples, replace=False)
        X_sample = X[indices]
        print(f"  Using {max_samples} samples (subsampled from {n_samples})")
    else:
        X_sample = X
    
    # Create explainer
    try:
        if model_type == 'tree':
            # TreeExplainer for tree-based models (faster)
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for any model (slower but universal)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x),
                shap.sample(X_sample, min(100, len(X_sample)))  # Use background sample
            )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        print(f"  ✅ SHAP computation successful")
        
    except Exception as e:
        print(f"  ⚠️  SHAP computation failed: {e}")
        explainer = None
        shap_values = None
    
    return explainer, shap_values, X_sample


def plot_shap_summary(explainer, shap_values, X_sample, feature_names, save_path, plot_type='bar'):
    """
    Plot SHAP summary (feature importance).
    
    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : array-like
        SHAP values. Can be:
        - (n_samples, n_features) for binary
        - (n_samples, n_features, n_classes) for multiclass
        - list of arrays for some models
    X_sample : array-like
        Feature matrix used for SHAP computation.
    feature_names : list of str
        Feature names.
    save_path : str
        Path to save figure.
    plot_type : str, default='bar'
        'bar' (feature importance) or 'beeswarm' (feature impact).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ─────────────────────────────────────────
        # Handle different SHAP value shapes
        # ─────────────────────────────────────────
        if isinstance(shap_values, list):
            # List of arrays (one per class for multiclass)
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes) → average across classes
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # (n_samples, n_features) → standard case
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        if plot_type == 'bar':
            # Sort by importance
            indices = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20
            
            ax.barh(range(len(indices)), mean_abs_shap[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title("Feature Importance (SHAP)", fontsize=12, fontweight='bold')
            ax.invert_yaxis()
        
        elif plot_type == 'beeswarm':
            # Fallback to bar if beeswarm issues
            indices = np.argsort(mean_abs_shap)[::-1][:20]
            ax.barh(range(len(indices)), mean_abs_shap[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title("Feature Importance (SHAP)", fontsize=12, fontweight='bold')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP summary saved to {save_path}")
        plt.close()
        
        return fig
        
    except Exception as e:
        print(f"⚠️  SHAP summary plot failed: {e}")
        return None


def plot_shap_force(explainer, shap_values, X_sample, sample_index, feature_names, save_path):
    """
    Plot SHAP force plot for a single sample (individual explanation).
    
    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : array-like
        SHAP values (can be multiclass).
    X_sample : array-like
        Feature matrix.
    sample_index : int
        Index of sample to explain.
    feature_names : list of str
        Feature names.
    save_path : str
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    
    try:
        # ─────────────────────────────────────────
        # Handle different SHAP value shapes
        # ─────────────────────────────────────────
        if isinstance(shap_values, list):
            sv = shap_values[0]  # Use first class
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes) → use first class
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values
        
        # Create force plot (manual version for compatibility)
        sample_shap = sv[sample_index]
        sample_features = X_sample[sample_index]
        
        # Get non-zero SHAP values
        nonzero_idx = np.nonzero(sample_shap)[0]
        if len(nonzero_idx) == 0:
            nonzero_idx = np.arange(len(sample_shap))
        
        nonzero_idx = nonzero_idx[np.argsort(np.abs(sample_shap[nonzero_idx]))[::-1][:10]]  # Top 10
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap_vals = sample_shap[nonzero_idx]
        feat_vals = sample_features[nonzero_idx]
        feat_names_subset = [feature_names[i] for i in nonzero_idx]
        
        colors = ['red' if v < 0 else 'blue' for v in shap_vals]
        ax.barh(range(len(nonzero_idx)), shap_vals, color=colors)
        ax.set_yticks(range(len(nonzero_idx)))
        ax.set_yticklabels([f"{name}\n({val:.3f})" for name, val in zip(feat_names_subset, feat_vals)])
        ax.set_xlabel("SHAP Value", fontsize=11)
        ax.set_title(f"Sample {sample_index}: Feature Contributions (Class 0)", fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP force plot saved to {save_path}")
        plt.close()
        
        return fig
        
    except Exception as e:
        print(f"⚠️  SHAP force plot failed: {e}")
        return None


def plot_shap_dependence(explainer, shap_values, X_sample, feature_idx, feature_names, save_path):
    """
    Plot SHAP dependence plot (feature interaction).
    
    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : array-like
        SHAP values (can be multiclass).
    X_sample : array-like
        Feature matrix.
    feature_idx : int
        Index of feature to analyze.
    feature_names : list of str
        Feature names.
    save_path : str
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    
    try:
        # ─────────────────────────────────────────
        # Handle different SHAP value shapes
        # ─────────────────────────────────────────
        if isinstance(shap_values, list):
            sv = shap_values[0]  # Use first class
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes) → use first class
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # SHAP dependence: feature value vs SHAP value
        feature_vals = X_sample[:, feature_idx]
        shap_vals = sv[:, feature_idx]
        
        ax.scatter(feature_vals, shap_vals, alpha=0.5, s=20, color='steelblue')
        ax.set_xlabel(f"{feature_names[feature_idx]} Value", fontsize=11)
        ax.set_ylabel(f"SHAP Value for {feature_names[feature_idx]}", fontsize=11)
        ax.set_title(f"SHAP Dependence Plot: {feature_names[feature_idx]} (Class 0)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP dependence plot saved to {save_path}")
        plt.close()
        
        return fig
        
    except Exception as e:
        print(f"⚠️  SHAP dependence plot failed: {e}")
        return None


def feature_importance_ranking(shap_values, feature_names, top_n=15):
    """
    Rank features by mean absolute SHAP value.
    
    Parameters
    ----------
    shap_values : array-like
        SHAP values. Can be:
        - (n_samples, n_features) for binary
        - (n_samples, n_features, n_classes) for multiclass
        - list of arrays
    feature_names : list of str
        Feature names.
    top_n : int, default=15
        Number of top features to return.
    
    Returns
    -------
    importance_df : pd.DataFrame
        Columns: ['Rank', 'Feature', 'Mean |SHAP|', 'Relative Importance (%)']
    """
    
    try:
        # Handle different SHAP value shapes
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes) → average across classes
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # (n_samples, n_features) → standard case
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort and rank
        indices = np.argsort(mean_abs_shap)[::-1]
        
        mean_abs_shap_sorted = mean_abs_shap[indices]
        total_importance = mean_abs_shap_sorted.sum()
        
        importance_data = {
            'Rank': np.arange(1, len(indices) + 1),
            'Feature': [feature_names[i] for i in indices],
            'Mean |SHAP|': mean_abs_shap_sorted,
            'Relative Importance (%)': 100 * mean_abs_shap_sorted / total_importance,
        }
        
        importance_df = pd.DataFrame(importance_data).head(top_n)
        
        return importance_df
        
    except Exception as e:
        print(f"⚠️  Feature importance ranking failed: {e}")
        return None