"""
Component 1️⃣1️⃣: Reporting and consolidated results.
Creates PDF/CSV summaries of all 11 components.
"""

import os
import pandas as pd
import json
from datetime import datetime


def save_metrics_tables(model_name, overall_dict, per_class_df, output_dir):
    """
    Save overall and per-class metrics to CSV.
    
    Parameters
    ----------
    model_name : str
        Name of model (e.g., "CatBoost").
    overall_dict : dict
        Dict with overall metrics (accuracy, macro_f1, etc.).
    per_class_df : pd.DataFrame
        DataFrame with per-class metrics.
    output_dir : str
        Where to save CSVs.
    
    Returns
    -------
    overall_path, per_class_path : tuple of str
        Paths to saved CSV files.
    """
    
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    
    # Save overall metrics
    overall_df = pd.DataFrame([overall_dict])
    overall_path = os.path.join(output_dir, "tables", f"overall_metrics_{model_name.lower()}.csv")
    overall_df.to_csv(overall_path, index=False)
    print(f"✅ Overall metrics saved: {overall_path}")
    
    # Save per-class metrics
    per_class_path = os.path.join(output_dir, "tables", f"per_class_metrics_{model_name.lower()}.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print(f"✅ Per-class metrics saved: {per_class_path}")
    
    return overall_path, per_class_path


def generate_consolidated_report(results_dict, model_name, output_dir):
    """
    Create a formatted CSV summary of all 11 components.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with all component outputs.
    model_name : str
        Name of model.
    output_dir : str
        Where to save report files.
    
    Returns
    -------
    report_path : str
        Path to saved consolidated report.
    """
    
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    # Build consolidated report
    report_sections = []
    
    # Header
    report_sections.append({
        'section': 'Header',
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metric': 'N/A',
        'value': 'N/A',
    })
    
    # 1️⃣ 2️⃣ Overall & Per-class Metrics
    if 'overall_metrics' in results_dict:
        for key, val in results_dict['overall_metrics'].items():
            report_sections.append({
                'section': 'Overall Metrics',
                'model_name': model_name,
                'timestamp': '',
                'metric': key,
                'value': f"{val:.4f}" if isinstance(val, float) else val,
            })
    
    # 6️⃣ Bootstrap CIs
    if 'ci_results' in results_dict:
        for metric_name, ci_info in results_dict['ci_results'].items():
            report_sections.append({
                'section': 'Bootstrap CI (95%)',
                'model_name': model_name,
                'timestamp': '',
                'metric': f"{metric_name}_mean",
                'value': f"{ci_info['mean']:.4f}",
            })
            report_sections.append({
                'section': 'Bootstrap CI (95%)',
                'model_name': model_name,
                'timestamp': '',
                'metric': f"{metric_name}_ci",
                'value': f"[{ci_info['ci_low']:.4f}, {ci_info['ci_high']:.4f}]",
            })
    
    # 8️⃣ Uncertainty
    if 'uncertainty_summary' in results_dict:
        unc_summary = results_dict['uncertainty_summary']
        report_sections.append({
            'section': 'Uncertainty Signals',
            'model_name': model_name,
            'timestamp': '',
            'metric': 'entropy_mean',
            'value': f"{unc_summary['entropy']['mean']:.4f}",
        })
        report_sections.append({
            'section': 'Uncertainty Signals',
            'model_name': model_name,
            'timestamp': '',
            'metric': 'margin_mean',
            'value': f"{unc_summary['margin']['mean']:.4f}",
        })
    
    # 7️⃣ Error Summary
    if 'error_summary' in results_dict:
        err_summary = results_dict['error_summary']
        report_sections.append({
            'section': 'Error Summary',
            'model_name': model_name,
            'timestamp': '',
            'metric': 'total_samples',
            'value': err_summary['total_samples'],
        })
        report_sections.append({
            'section': 'Error Summary',
            'model_name': model_name,
            'timestamp': '',
            'metric': 'error_rate',
            'value': f"{err_summary['error_rate']:.4f}",
        })
    
    # 1️⃣0️⃣ Bias Diagnostics
    if 'disparate_impact_ratio' in results_dict:
        report_sections.append({
            'section': 'Bias Diagnostics',
            'model_name': model_name,
            'timestamp': '',
            'metric': 'disparate_impact_ratio',
            'value': f"{results_dict['disparate_impact_ratio']:.4f}",
        })
    
    # Create DataFrame and save
    report_df = pd.DataFrame(report_sections)
    report_path = os.path.join(output_dir, "reports", f"consolidated_report_{model_name.lower()}.csv")
    report_df.to_csv(report_path, index=False)
    print(f"✅ Consolidated report saved: {report_path}")
    
    return report_path


def generate_json_summary(results_dict, model_name, output_dir):
    """
    Export complete results as JSON for downstream use.
    
    Parameters
    ----------
    results_dict : dict
        All component results.
    model_name : str
        Model name.
    output_dir : str
        Output directory.
    
    Returns
    -------
    json_path : str
        Path to saved JSON file.
    """
    
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    # Prepare JSON-serializable dict
    json_dict = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add serializable results
    for key, val in results_dict.items():
        if isinstance(val, (dict, list, str, int, float)):
            json_dict[key] = val
        elif isinstance(val, pd.DataFrame):
            json_dict[key] = val.to_dict(orient='records')
        else:
            json_dict[key] = str(val)
    
    json_path = os.path.join(output_dir, "reports", f"results_{model_name.lower()}.json")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2, default=str)
    
    print(f"✅ JSON summary saved: {json_path}")
    
    return json_path


def print_summary_table(results_dict, model_name):
    """
    Print a nice summary table to console.
    
    Parameters
    ----------
    results_dict : dict
        All component results.
    model_name : str
        Model name.
    """
    
    print("\n" + "=" * 80)
    print(f"📊 EVALUATION SUMMARY: {model_name}")
    print("=" * 80)
    
    if 'overall_metrics' in results_dict:
        print("\n🎯 OVERALL METRICS:")
        for key, val in results_dict['overall_metrics'].items():
            if isinstance(val, float):
                print(f"  {key:.<30} {val:.4f}")
            else:
                print(f"  {key:.<30} {val}")
    
    if 'error_summary' in results_dict:
        err = results_dict['error_summary']
        print("\n📈 ERROR ANALYSIS:")
        print(f"  Total samples:................. {err['total_samples']}")
        print(f"  Correct predictions:.......... {err['correct_predictions']}")
        print(f"  Error predictions:........... {err['error_predictions']}")
        print(f"  Error rate:................... {err['error_rate']:.2%}")
    
    if 'uncertainty_summary' in results_dict:
        unc = results_dict['uncertainty_summary']
        print("\n🎲 UNCERTAINTY SIGNALS:")
        print(f"  Entropy (mean):............... {unc['entropy']['mean']:.4f}")
        print(f"  Margin (mean):............... {unc['margin']['mean']:.4f}")
        print(f"  Max probability (mean):...... {unc['max_prob']['mean']:.4f}")
    
    if 'ci_results' in results_dict:
        print("\n📊 BOOTSTRAP CONFIDENCE INTERVALS (95%):")
        ci_res = results_dict['ci_results']
        if 'accuracy' in ci_res:
            acc_ci = ci_res['accuracy']
            print(f"  Accuracy CI:.................. [{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")
        if 'macro_f1' in ci_res:
            f1_ci = ci_res['macro_f1']
            print(f"  Macro F1 CI:.................. [{f1_ci['ci_low']:.4f}, {f1_ci['ci_high']:.4f}]")
    
    if 'disparate_impact_ratio' in results_dict:
        print("\n⚖️ BIAS DIAGNOSTICS:")
        di = results_dict['disparate_impact_ratio']
        status = "✅ PASS" if di >= 0.8 else "⚠️ FAIL"
        print(f"  Disparate impact ratio:....... {di:.4f} {status}")
    
    print("\n" + "=" * 80 + "\n")