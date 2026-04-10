"""
Test Component 6️⃣: Bootstrap confidence intervals.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import statistical_inference
from utils import constants


def test_bootstrap_confidence_intervals():
    """Test bootstrap CI computation."""
    print("\n" + "=" * 80)
    print("TEST: bootstrap_confidence_intervals()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    # Create synthetic data
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
    
    # Create probabilities
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Compute bootstrap CIs (use fewer bootstraps for faster testing)
    print("\nComputing bootstrap CIs (n_bootstrap=100 for testing)...")
    ci_results, bootstrap_samples = statistical_inference.bootstrap_confidence_intervals(
        y_true,
        y_pred,
        y_prob,
        constants.CLASS_NAMES,
        n_bootstrap=100,
        ci=95,
        random_state=constants.RANDOM_SEED,
    )
    
    print("\n📊 BOOTSTRAP CONFIDENCE INTERVALS (Component 6️⃣)")
    print("-" * 80)
    
    print("\n🎯 OVERALL METRICS:")
    for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
        if metric in ci_results:
            result = ci_results[metric]
            print(f"  {metric:.<30}")
            print(f"    Mean:    {result['mean']:.4f}")
            print(f"    95% CI:  [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
            print(f"    Std:     {result['std']:.4f}")
    
    print("\n📊 PER-CLASS METRICS (F1 only for brevity):")
    for class_name in constants.CLASS_NAMES:
        metric_key = f'{class_name}_f1'
        if metric_key in ci_results:
            result = ci_results[metric_key]
            print(f"  {class_name} F1:            [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    
    # ─────────────────────────────────────────
    # ✅ ASSERTIONS
    # ─────────────────────────────────────────
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(ci_results, dict), "Should return dict of CIs"
    assert isinstance(bootstrap_samples, dict), "Should return bootstrap samples"
    print("  ✅ Return types correct")
    
    # Check CI structure
    for metric_name, result in ci_results.items():
        assert isinstance(result, dict), f"{metric_name} result should be dict"
        assert set(result.keys()) == {'mean', 'ci_low', 'ci_high', 'std'}, \
            f"{metric_name} should have mean, ci_low, ci_high, std"
        assert result['ci_low'] <= result['mean'] <= result['ci_high'], \
            f"CI bounds should bracket mean for {metric_name}"
    print("  ✅ CI structure valid (ci_low <= mean <= ci_high)")
    
    # Check metric ranges
    for metric_name in ['accuracy', 'macro_f1']:
        assert 0 <= ci_results[metric_name]['mean'] <= 1, \
            f"{metric_name} mean should be in [0, 1]"
    print("  ✅ Metric values in valid ranges [0, 1]")
    
    # Check bootstrap samples
    assert len(bootstrap_samples['accuracy']) == 100, "Should have 100 bootstrap samples"
    print("  ✅ Bootstrap samples stored correctly")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: bootstrap_confidence_intervals() Working Correctly!")
    print("=" * 80 + "\n")


def test_bootstrap_ci_table():
    """Test CI table generation."""
    print("\n" + "=" * 80)
    print("TEST: bootstrap_ci_table()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
    
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Compute bootstrap CIs
    ci_results, _ = statistical_inference.bootstrap_confidence_intervals(
        y_true, y_pred, y_prob, constants.CLASS_NAMES, n_bootstrap=100, random_state=constants.RANDOM_SEED
    )
    
    # Generate CI table
    ci_df = statistical_inference.bootstrap_ci_table(ci_results, constants.CLASS_NAMES)
    
    print("\n📊 BOOTSTRAP CI TABLE")
    print("-" * 80)
    print(ci_df.to_string(index=False))
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(ci_df, pd.DataFrame), "Should return DataFrame"
    assert set(ci_df.columns) == {'Metric', 'Mean', 'CI Low', 'CI High', 'Std'}, \
        "Should have correct columns"
    print("  ✅ DataFrame structure correct")
    
    assert len(ci_df) > 0, "DataFrame should have rows"
    print(f"  ✅ DataFrame has {len(ci_df)} metrics")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: bootstrap_ci_table() Working Correctly!")
    print("=" * 80 + "\n")


def test_ci_width_summary():
    """Test CI width summary statistics."""
    print("\n" + "=" * 80)
    print("TEST: ci_width_summary()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
    
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Compute bootstrap CIs
    ci_results, _ = statistical_inference.bootstrap_confidence_intervals(
        y_true, y_pred, y_prob, constants.CLASS_NAMES, n_bootstrap=100, random_state=constants.RANDOM_SEED
    )
    
    # Get CI width summary
    summary = statistical_inference.ci_width_summary(ci_results)
    
    print("\n📊 CI WIDTH SUMMARY (Metric Stability)")
    print("-" * 80)
    for stat_name, value in summary.items():
        print(f"  {stat_name:.<30} {value:.4f}")
    
    print("\n  ℹ️  Smaller CI widths indicate more stable metrics across resamples.")
    print("  ℹ️  Larger CI widths suggest high variability in the test set.")
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(summary, dict), "Should return dict"
    assert set(summary.keys()) == {'mean_ci_width', 'median_ci_width', 'max_ci_width', 'min_ci_width'}, \
        "Should have all width statistics"
    print("  ✅ Summary structure correct")
    
    assert summary['min_ci_width'] <= summary['median_ci_width'] <= summary['max_ci_width'], \
        "Should have min <= median <= max"
    print("  ✅ CI widths ordered correctly")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: ci_width_summary() Working Correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_bootstrap_confidence_intervals()
    test_bootstrap_ci_table()
    test_ci_width_summary()
    print("\n🎉 ALL BOOTSTRAP CI TESTS COMPLETED SUCCESSFULLY! 🎉\n")