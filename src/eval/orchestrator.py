"""
🎯 MASTER ORCHESTRATOR: Chains all 11 components for complete model evaluation.
Entry point for comprehensive evaluation pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import all 11 component modules
from . import metrics
from . import visualizations
from . import uncertainty
from . import statistical_inference
from . import error_taxonomy
from . import explainability
from . import bias_diagnostics
from . import reporting


def analyze_model_performance(
    y_true,
    y_pred,
    y_prob,
    model,
    X_test,
    class_names,
    model_name,
    feature_names=None,
    output_dir="../reports",
    sample_ids=None,
    metadata_df=None,
    demographic_col=None,
):
    """
    🎯 MASTER ORCHESTRATOR: Chains all 11 evaluation components.
    
    This function orchestrates the complete evaluation pipeline:
    1️⃣  Classification Metrics (accuracy, F1, precision, recall)
    2️⃣  Per-Class Metrics Table
    3️⃣  Confusion Matrix Visualization
    4️⃣  ROC Curves (One-vs-Rest)
    5️⃣  Calibration Curves
    6️⃣  Bootstrap Confidence Intervals (95% CI)
    7️⃣  Error Taxonomy Export
    8️⃣  Uncertainty Signals (entropy, margin, confidence)
    9️⃣  SHAP Explainability (feature importance, dependence plots)
    1️⃣0️⃣ Bias Diagnostics (demographic parity, disparate impact)
    1️⃣1️⃣ Reporting & Consolidation (CSV, JSON, summary tables)
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels (numeric 0..C-1).
    y_pred : array-like, shape (n_samples,)
        Predicted class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted class probabilities.
    model : sklearn estimator
        Trained model with predict/predict_proba methods.
    X_test : array-like, shape (n_samples, n_features)
        Test feature matrix.
    class_names : list of str
        Human-readable class names (e.g., ["SCD", "MCI", "AD"]).
    model_name : str
        Model identifier (e.g., "CatBoost", "RandomForest").
    feature_names : list of str, optional
        Feature names for SHAP analysis. 
        Default: auto-generated as Feature_0, Feature_1, ...
    output_dir : str, default="../reports"
        Root directory for saving all outputs (figures, tables, reports).
    sample_ids : array-like, optional
        Sample identifiers (e.g., patient RID). Default: 0-indexed integers.
    metadata_df : pd.DataFrame, optional
        Sample metadata (age, sex, scanner, etc.).
        Must have same length as y_true.
    demographic_col : str, optional
        Column name for demographic group analysis in bias diagnostics.
        Example: 'sex', 'race', 'scanner'. Must be in metadata_df.
    
    Returns
    -------
    summary_dict : dict
        Compact summary with key metrics for model comparison:
        - model_name: str
        - accuracy: float (0-1)
        - macro_f1: float (0-1)
        - error_rate: float (0-1)
        - disparate_impact: float (0-1) or NaN
        - accuracy_ci: str "[lower, upper]"
        - macro_f1_ci: str "[lower, upper]"
    
    Raises
    ------
    ValueError
        If shapes of y_true, y_pred, y_prob don't match.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> 
    >>> # Prepare data
    >>> X_test = np.random.randn(200, 9)
    >>> y_test = np.random.randint(0, 3, 200)
    >>> model = RandomForestClassifier().fit(X_test, y_test)
    >>> y_pred = model.predict(X_test)
    >>> y_prob = model.predict_proba(X_test)
    >>> 
    >>> # Run orchestrator
    >>> summary = analyze_model_performance(
    ...     y_true=y_test,
    ...     y_pred=y_pred,
    ...     y_prob=y_prob,
    ...     model=model,
    ...     X_test=X_test,
    ...     class_names=["SCD", "MCI", "AD"],
    ...     model_name="RandomForest",
    ...     demographic_col='sex',
    ... )
    >>> print(summary['accuracy'])
    0.85
    """
    
    # ─────────────────────────────────────────
    # SETUP & VALIDATION
    # ─────────────────────────────────────────
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    X_test = np.array(X_test)
    
    n_samples = len(y_true)
    n_classes = len(class_names)
    n_features = X_test.shape[1]
    
    # Validation
    if len(y_pred) != n_samples:
        raise ValueError(f"y_pred length {len(y_pred)} != y_true length {n_samples}")
    if y_prob.shape != (n_samples, n_classes):
        raise ValueError(f"y_prob shape {y_prob.shape} != ({n_samples}, {n_classes})")
    
    # Default feature names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    # Storage for all results
    results = {}
    
    start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"🎯 MASTER ORCHESTRATOR: {model_name}")
    print(f"{'='*80}")
    print(f"Test samples: {n_samples}")
    print(f"Classes: {n_classes} {class_names}")
    print(f"Features: {n_features}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────
    # COMPONENT 1️⃣ 2️⃣: CLASSIFICATION METRICS
    # ─────────────────────────────────────────
    print("[1️⃣ 2️⃣] Computing classification metrics...")
    try:
        overall_metrics, per_class_df = metrics.compute_classification_metrics(
            y_true, y_pred, y_prob, class_names
        )
        results['overall_metrics'] = overall_metrics
        results['per_class_metrics'] = per_class_df
        
        print(f"   ✅ Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"   ✅ Macro F1: {overall_metrics['macro_f1']:.4f}")
        print(f"   ✅ Weighted F1: {overall_metrics['weighted_f1']:.4f}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise
    
    # ─────────────────────────────────────────
    # COMPONENT 3️⃣ 4️⃣ 5️⃣: VISUALIZATIONS
    # ─────────────────────────────────────────
    print("\n[3️⃣ 4️⃣ 5️⃣] Generating visualizations...")
    try:
        # Confusion matrix
        cm_path = os.path.join(output_dir, "figures", f"confusion_matrix_{model_name.lower()}.png")
        visualizations.plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        print(f"   ✅ Confusion matrix: {cm_path}")
        
        # ROC curves
        roc_path = os.path.join(output_dir, "figures", f"roc_curves_{model_name.lower()}.png")
        visualizations.plot_roc_ovr(y_true, y_prob, class_names, roc_path)
        print(f"   ✅ ROC curves: {roc_path}")
        
        # Calibration curves
        cal_path = os.path.join(output_dir, "figures", f"calibration_curves_{model_name.lower()}.png")
        visualizations.plot_calibration_curve(y_true, y_prob, class_names, cal_path)
        print(f"   ✅ Calibration curves: {cal_path}")
    except Exception as e:
        print(f"   ⚠️  Visualization warning: {e}")
    
    # ─────────────────────────────────────────
    # COMPONENT 8️⃣: UNCERTAINTY SIGNALS
    # ─────────────────────────────────────────
    print("\n[8️⃣] Computing uncertainty signals...")
    try:
        uncertainty_df = uncertainty.compute_uncertainty_signals(y_prob)
        unc_summary = uncertainty.uncertainty_summary_stats(uncertainty_df)
        results['uncertainty'] = uncertainty_df
        results['uncertainty_summary'] = unc_summary
        
        print(f"   ✅ Entropy (mean): {unc_summary['entropy']['mean']:.4f}")
        print(f"   ✅ Margin (mean): {unc_summary['margin']['mean']:.4f}")
        print(f"   ✅ Max prob (mean): {unc_summary['max_prob']['mean']:.4f}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # ─────────────────────────────────────────
    # COMPONENT 6️⃣: BOOTSTRAP CONFIDENCE INTERVALS
    # ─────────────────────────────────────────
    print("\n[6️⃣] Computing bootstrap confidence intervals (95%)...")
    try:
        ci_results, bootstrap_samples = statistical_inference.bootstrap_confidence_intervals(
            y_true, y_pred, y_prob, class_names,
            n_bootstrap=100,  # Use 100 for speed; increase to 1000 for final
            ci=95,
        )
        results['ci_results'] = ci_results
        
        ci_df = statistical_inference.bootstrap_ci_table(ci_results, class_names)
        ci_path = os.path.join(output_dir, "tables", f"bootstrap_ci_{model_name.lower()}.csv")
        ci_df.to_csv(ci_path, index=False)
        
        print(f"   ✅ 95% CIs computed for {len(ci_results)} metrics")
        
        # Show key CIs
        if 'accuracy' in ci_results:
            acc_ci = ci_results['accuracy']
            print(f"   ✅ Accuracy CI: [{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")
    except Exception as e:
        print(f"   ⚠️  Bootstrap warning: {e}")
    
    # ─────────────────────────────────────────
    # COMPONENT 7️⃣: ERROR TAXONOMY
    # ─────────────────────────────────────────
    print("\n[7️⃣] Exporting error taxonomy...")
    try:
        error_df = error_taxonomy.export_error_taxonomy(
            y_true, y_pred, y_prob, class_names,
            model_name=model_name,
            save_path=os.path.join(output_dir, "tables", f"error_taxonomy_{model_name.lower()}.csv"),
            sample_ids=sample_ids,
            metadata_df=metadata_df,
        )
        results['error_taxonomy'] = error_df
        
        # Error summary statistics
        error_summary = error_taxonomy.error_taxonomy_summary(error_df, class_names)
        results['error_summary'] = error_summary
        
        # HITL review cases (top uncertain errors)
        top_uncertain = error_taxonomy.error_cases_by_uncertainty(error_df, n_top=10)
        hitl_path = os.path.join(output_dir, "tables", f"hitl_review_{model_name.lower()}.csv")
        top_uncertain.to_csv(hitl_path, index=False)
        
        print(f"   ✅ Error taxonomy: {len(error_df)} samples")
        print(f"   ✅ Error rate: {error_summary['error_rate']:.2%}")
        print(f"   ✅ HITL review: {hitl_path}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise
    
    # ─────────────────────────────────────────
    # COMPONENT 9️⃣: SHAP EXPLAINABILITY
    # ─────────────────────────────────────────
    print("\n[9️⃣] Computing SHAP explanations...")
    try:
        explainer, shap_values, X_sample = explainability.compute_shap_values(
            model, X_test, class_names, model_type='tree', max_samples=100
        )
        
        # Summary plot
        explainability.plot_shap_summary(
            explainer, shap_values, X_sample, feature_names,
            os.path.join(output_dir, "figures", f"shap_summary_{model_name.lower()}.png"),
            plot_type='bar'
        )
        print(f"   ✅ SHAP summary plot saved")
        
        # Feature importance ranking
        importance_df = explainability.feature_importance_ranking(
            shap_values, feature_names, top_n=15
        )
        results['feature_importance'] = importance_df
        importance_path = os.path.join(output_dir, "tables", f"feature_importance_{model_name.lower()}.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"   ✅ Feature importance: {importance_path}")
        
        # Sample-level explanation
        explainability.plot_shap_force(
            explainer, shap_values, X_sample, sample_index=0, feature_names=feature_names,
            save_path=os.path.join(output_dir, "figures", f"shap_force_sample0_{model_name.lower()}.png")
        )
        print(f"   ✅ Sample explanation saved")
        
    except Exception as e:
        print(f"   ⚠️  SHAP warning: {e}")
    
    # ─────────────────────────────────────────
    # COMPONENT 1️⃣0️⃣: BIAS DIAGNOSTICS
    # ─────────────────────────────────────────
    print("\n[1️⃣0️⃣] Running bias diagnostics...")
    try:
        if demographic_col and demographic_col in error_df.columns:
            # Demographic parity
            parity_df = bias_diagnostics.demographic_parity(
                error_df, demographic_col
            )
            results['demographic_parity'] = parity_df
            parity_path = os.path.join(output_dir, "tables", f"demographic_parity_{model_name.lower()}.csv")
            parity_df.to_csv(parity_path, index=False)
            
            # Fairness metrics by group
            metrics_df = bias_diagnostics.fairness_metrics_by_group(
                error_df, demographic_col, class_names
            )
            results['fairness_metrics'] = metrics_df
            
            # Disparate impact
            impact_df, di_ratio = bias_diagnostics.disparate_impact(
                error_df, demographic_col
            )
            results['disparate_impact_ratio'] = di_ratio
            
            print(f"   ✅ Demographic parity: {parity_path}")
            print(f"   ✅ Disparate impact ratio: {di_ratio:.4f}", end="")
            if di_ratio >= 0.8:
                print(" ✅ PASS (>= 0.8 rule)")
            else:
                print(" ⚠️ WARNING (< 0.8)")
        else:
            print(f"   ⚠️  Demographic column '{demographic_col}' not found")
    except Exception as e:
        print(f"   ⚠️  Bias diagnostics warning: {e}")
    
    # ─────────────────────────────────────────
    # COMPONENT 1️⃣1️⃣: REPORTING & CONSOLIDATION
    # ─────────────────────────────────────────
    print("\n[1️⃣1️⃣] Generating reports...")
    try:
        # Save metrics tables
        reporting.save_metrics_tables(
            model_name, overall_metrics, per_class_df, output_dir
        )
        
        # Consolidated report
        reporting.generate_consolidated_report(results, model_name, output_dir)
        
        # JSON summary
        reporting.generate_json_summary(results, model_name, output_dir)
        
        # Print summary table to console
        reporting.print_summary_table(results, model_name)
        
    except Exception as e:
        print(f"   ⚠️  Reporting warning: {e}")
    
    # ─────────────────────────────────────────
    # FINALIZE & BUILD SUMMARY DICT
    # ─────────────────────────────────────────
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"✅ ORCHESTRATOR COMPLETED in {elapsed_time:.1f}s")
    print(f"{'='*80}\n")
    
    # Compact summary for model comparison
    summary_dict = {
        'model_name': model_name,
        'accuracy': overall_metrics['accuracy'],
        'macro_f1': overall_metrics['macro_f1'],
        'weighted_f1': overall_metrics['weighted_f1'],
        'error_rate': error_summary['error_rate'],
        'elapsed_time_s': elapsed_time,
    }
    
    # Add CIs if available
    if 'ci_results' in results and results['ci_results']:
        if 'accuracy' in results['ci_results']:
            acc_ci = results['ci_results']['accuracy']
            summary_dict['accuracy_ci'] = f"[{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]"
        if 'macro_f1' in results['ci_results']:
            f1_ci = results['ci_results']['macro_f1']
            summary_dict['macro_f1_ci'] = f"[{f1_ci['ci_low']:.4f}, {f1_ci['ci_high']:.4f}]"
    
    # Add disparate impact if available
    if 'disparate_impact_ratio' in results:
        summary_dict['disparate_impact'] = results['disparate_impact_ratio']
    
    return summary_dict


def compare_models(model_results_list, output_dir="../reports"):
    """
    Compare multiple models based on orchestrator outputs.
    
    Parameters
    ----------
    model_results_list : list of dict
        List of summary dictionaries from analyze_model_performance().
    output_dir : str
        Directory to save comparison.
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Model comparison table.
    """
    
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    comparison_df = pd.DataFrame(model_results_list)
    
    print(comparison_df.to_string(index=False))
    
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    comparison_path = os.path.join(output_dir, "reports", "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n✅ Model comparison saved: {comparison_path}\n")
    
    return comparison_df