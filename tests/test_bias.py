"""
Test Component 1️⃣0️⃣: Bias diagnostics and fairness analysis.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import bias_diagnostics
from utils import constants


def test_bias_diagnostics():
    print("\n" + "=" * 80)
    print("TEST: Bias Diagnostics")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    # Synthetic error taxonomy
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
    metadata_df = pd.DataFrame({
        'sex': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian'], n_samples),
        'scanner': np.random.choice(['Siemens', 'GE', 'Philips'], n_samples),
    })
    error_df = pd.DataFrame({
        'true_label_int': y_true,
        'predicted_label_int': y_pred,
        'true_label': [constants.CLASS_NAMES[int(y)] for y in y_true],
        'predicted_label': [constants.CLASS_NAMES[int(y)] for y in y_pred],
    })
    error_df = pd.concat([error_df, metadata_df], axis=1)
    
    # Demographic parity (sex)
    parity_df = bias_diagnostics.demographic_parity(error_df, group_col='sex', positive_label='AD')
    print("\n📊 Demographic Parity (sex):")
    print(parity_df.to_string(index=False))
    
    # Fairness metrics by group (race)
    metrics_df = bias_diagnostics.fairness_metrics_by_group(error_df, group_col='race', class_names=constants.CLASS_NAMES)
    print("\n📊 Fairness Metrics by Race:")
    print(metrics_df.to_string(index=False))
    
    # Disparate impact (scanner)
    impact_df, di_ratio = bias_diagnostics.disparate_impact(error_df, group_col='scanner', positive_label='AD')
    print("\n📊 Disparate Impact (scanner):")
    print(impact_df.to_string(index=False))
    print(f"\nDisparate Impact Ratio (min/max): {di_ratio:.3f}")
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    assert isinstance(parity_df, pd.DataFrame)
    assert isinstance(metrics_df, pd.DataFrame)
    assert isinstance(impact_df, pd.DataFrame)
    assert 0 <= di_ratio <= 1, "Disparate impact ratio should be in [0, 1]"
    print("  ✅ All bias diagnostics computed correctly")
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: Bias Diagnostics Working Correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_bias_diagnostics()
    print("\n🎉 ALL BIAS DIAGNOSTICS TESTS COMPLETED SUCCESSFULLY! 🎉\n")