"""
Chains all 11 components for complete model evaluation.
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
    Run the full 11-component evaluation pipeline for a single model.
    """

    # ----------------------------------------------------------------------
    # SETUP & VALIDATION
    # ----------------------------------------------------------------------
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    y_prob = np.array(y_prob)
    X_test = np.array(X_test)

    n_samples = len(y_true)
    n_classes = len(class_names)
    n_features = X_test.shape[1]

    if len(y_pred) != n_samples:
        raise ValueError("y_pred length mismatch")
    if y_prob.shape != (n_samples, n_classes):
        raise ValueError("y_prob shape mismatch")

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Output dirs
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

    results = {}
    start_time = datetime.now()

    print("\n" + "="*80)
    print(f"Evaluating: {model_name}")
    print("="*80)
    print(f"Test samples: {n_samples}")
    print(f"Classes: {class_names}")
    print(f"Features: {n_features}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # ----------------------------------------------------------------------
    # 1 / 2: Classification metrics
    # ----------------------------------------------------------------------
    print("[1/2] Computing classification metrics...")

    overall_metrics, per_class_df = metrics.compute_classification_metrics(
        y_true, y_pred, y_prob, class_names
    )
    results["overall_metrics"] = overall_metrics
    results["per_class_metrics"] = per_class_df

    print(f"   Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"   Macro F1: {overall_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1: {overall_metrics['weighted_f1']:.4f}")

    # ----------------------------------------------------------------------
    # 3 / 4 / 5: Visualizations
    # ----------------------------------------------------------------------
    print("\n[3/4/5] Generating visualizations...")

    try:
        cm_path = os.path.join(output_dir, "figures", f"confusion_matrix_{model_name.lower()}.png")
        visualizations.plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        print(f"   Confusion matrix: {cm_path}")

        roc_path = os.path.join(output_dir, "figures", f"roc_curves_{model_name.lower()}.png")
        visualizations.plot_roc_ovr(y_true, y_prob, class_names, roc_path)
        print(f"   ROC curves: {roc_path}")

        cal_path = os.path.join(output_dir, "figures", f"calibration_curves_{model_name.lower()}.png")
        visualizations.plot_calibration_curve(y_true, y_prob, class_names, cal_path)
        print(f"   Calibration curves: {cal_path}")
    except Exception as e:
        print(f"   Visualization warning: {e}")

    # ----------------------------------------------------------------------
    # 8: Uncertainty
    # ----------------------------------------------------------------------
    print("\n[8] Computing uncertainty signals...")

    uncertainty_df = uncertainty.compute_uncertainty_signals(y_prob)
    unc_summary = uncertainty.uncertainty_summary_stats(uncertainty_df)

    results["uncertainty"] = uncertainty_df
    results["uncertainty_summary"] = unc_summary

    print(f"   Entropy (mean): {unc_summary['entropy']['mean']:.4f}")
    print(f"   Margin  (mean): {unc_summary['margin']['mean']:.4f}")
    print(f"   MaxProb (mean): {unc_summary['max_prob']['mean']:.4f}")

    # ----------------------------------------------------------------------
    # 6: Bootstrap Confidence Intervals
    # ----------------------------------------------------------------------
    print("\n[6] Computing bootstrap confidence intervals...")

    try:
        ci_results, boot = statistical_inference.bootstrap_confidence_intervals(
            y_true, y_pred, y_prob, class_names,
            n_bootstrap=100,
            ci=95
        )

        results["ci_results"] = ci_results

        ci_df = statistical_inference.bootstrap_ci_table(ci_results, class_names)
        ci_path = os.path.join(output_dir, "tables", f"bootstrap_ci_{model_name.lower()}.csv")
        ci_df.to_csv(ci_path, index=False)

        print(f"   CI table saved -> {ci_path}")

        if "accuracy" in ci_results:
            acc_ci = ci_results["accuracy"]
            print(f"   Accuracy CI: [{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")

    except Exception as e:
        print(f"   Bootstrap warning: {e}")

    # ----------------------------------------------------------------------
    # 7: Error Taxonomy
    # ----------------------------------------------------------------------
    print("\n[7] Exporting error taxonomy...")

    try:
        error_df = error_taxonomy.export_error_taxonomy(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
            model_name=model_name,
            save_path=os.path.join(output_dir, "tables", f"error_taxonomy_{model_name.lower()}.csv"),
            sample_ids=sample_ids,
            metadata_df=metadata_df,
        )

        results["error_taxonomy"] = error_df
        err_summary = error_taxonomy.error_taxonomy_summary(error_df, class_names)
        results["error_summary"] = err_summary

        top_unc = error_taxonomy.error_cases_by_uncertainty(error_df, n_top=10)
        hitl_path = os.path.join(output_dir, "tables", f"hitl_review_{model_name.lower()}.csv")
        top_unc.to_csv(hitl_path, index=False)

        print(f"   Error taxonomy: {len(error_df)} samples")
        print(f"   Error rate: {err_summary['error_rate']:.2%}")
        print(f"   HITL review: {hitl_path}")

    except Exception as e:
        print(f"   ERROR in error taxonomy: {e}")
        raise

    # ----------------------------------------------------------------------
    # 9: SHAP Explainability
    # ----------------------------------------------------------------------
    print("\n[9] Computing SHAP explanations...")

    try:
        # Only CatBoost has stable SHAP in this environment
        if model_name != "CatBoost":
            print("   SHAP disabled for non-CatBoost models (skipping)")
        else:
            explainer, shap_values, X_sample = explainability.compute_shap_values(
                model, X_test, class_names,
                model_type="tree",
                max_samples=100
            )

            shap_path = os.path.join(output_dir, "figures", f"shap_summary_{model_name.lower()}.png")
            explainability.plot_shap_summary(
                explainer, shap_values, X_sample,
                feature_names, shap_path, plot_type="bar"
            )
            print(f"   SHAP summary saved -> {shap_path}")

            imp_df = explainability.feature_importance_ranking(shap_values, feature_names, top_n=15)
            results["feature_importance"] = imp_df
            imp_path = os.path.join(output_dir, "tables", f"feature_importance_{model_name.lower()}.csv")
            imp_df.to_csv(imp_path, index=False)
            print(f"   Feature importance saved -> {imp_path}")

    except Exception as e:
        print(f"   SHAP warning: {e}")

    # ----------------------------------------------------------------------
    # 10: Bias Diagnostics
    # ----------------------------------------------------------------------
    print("\n[10] Running bias diagnostics...")

    try:
        if demographic_col and metadata_df is not None and demographic_col in metadata_df.columns:

            parity_df = bias_diagnostics.demographic_parity(error_df, demographic_col)
            results["demographic_parity"] = parity_df
            parity_path = os.path.join(output_dir, "tables", f"demographic_parity_{model_name.lower()}.csv")
            parity_df.to_csv(parity_path, index=False)

            impact_df, di_ratio = bias_diagnostics.disparate_impact(error_df, demographic_col)
            results["disparate_impact_ratio"] = di_ratio

            print(f"   Demographic parity saved -> {parity_path}")
            print(f"   Disparate impact ratio: {di_ratio:.4f}")

        else:
            print(f"   Demographic column '{demographic_col}' unavailable -- skipping bias tests.")

    except Exception as e:
        print(f"   Bias warning: {e}")

    # ----------------------------------------------------------------------
    # 11: Reporting
    # ----------------------------------------------------------------------
    print("\n[11] Generating reports...")

    try:
        reporting.save_metrics_tables(model_name, overall_metrics, per_class_df, output_dir)
        reporting.generate_consolidated_report(results, model_name, output_dir)
        reporting.generate_json_summary(results, model_name, output_dir)
        reporting.print_summary_table(results, model_name)
    except Exception as e:
        print(f"   Reporting warning: {e}")

    # ----------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*80)
    print(f"Evaluation complete for {model_name} in {elapsed:.2f}s")
    print("="*80)

    # Return compact summary dictionary
    summary = {
        "model_name": model_name,
        "accuracy": overall_metrics["accuracy"],
        "macro_f1": overall_metrics["macro_f1"],
        "weighted_f1": overall_metrics["weighted_f1"],
        "error_rate": err_summary["error_rate"],
        "elapsed_time_s": elapsed,
    }

    # Add accuracy CI if available
    if "ci_results" in results and "accuracy" in results["ci_results"]:
        ci = results["ci_results"]["accuracy"]
        summary["accuracy_ci"] = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"

    return summary
