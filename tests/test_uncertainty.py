"""
Test Component 8️⃣: Uncertainty signals (entropy, margin, confidence).
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from eval import uncertainty
from utils import constants


def test_compute_uncertainty_signals():
    """Test uncertainty signal computation."""
    print("\n" + "=" * 80)
    print("TEST: compute_uncertainty_signals()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    # Create synthetic probabilities
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Compute uncertainty signals
    uncertainty_df = uncertainty.compute_uncertainty_signals(y_prob)
    
    print("\n📊 UNCERTAINTY SIGNALS (Component 8️⃣)")
    print("-" * 80)
    print(uncertainty_df.head(10).to_string(index=False))
    
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 80)
    print(f"  Entropy:")
    print(f"    Mean: {uncertainty_df['entropy'].mean():.4f}")
    print(f"    Std:  {uncertainty_df['entropy'].std():.4f}")
    print(f"    Range: [{uncertainty_df['entropy'].min():.4f}, {uncertainty_df['entropy'].max():.4f}]")
    
    print(f"\n  Margin:")
    print(f"    Mean: {uncertainty_df['margin'].mean():.4f}")
    print(f"    Std:  {uncertainty_df['margin'].std():.4f}")
    print(f"    Range: [{uncertainty_df['margin'].min():.4f}, {uncertainty_df['margin'].max():.4f}]")
    
    print(f"\n  Max Probability:")
    print(f"    Mean: {uncertainty_df['max_prob'].mean():.4f}")
    print(f"    Std:  {uncertainty_df['max_prob'].std():.4f}")
    print(f"    Range: [{uncertainty_df['max_prob'].min():.4f}, {uncertainty_df['max_prob'].max():.4f}]")
    
    # ─────────────────────────────────────────
    # ✅ ASSERTIONS
    # ─────────────────────────────────────────
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(uncertainty_df, pd.DataFrame), "Should return DataFrame"
    assert uncertainty_df.shape[0] == n_samples, f"Should have {n_samples} rows"
    assert set(uncertainty_df.columns) == {'entropy', 'margin', 'max_prob', 'confidence_gap'}, \
        "Should have correct columns"
    print("  ✅ DataFrame shape and columns correct")
    
    assert (uncertainty_df['entropy'] >= 0).all(), "Entropy should be >= 0"
    assert (uncertainty_df['margin'] >= 0).all() and (uncertainty_df['margin'] <= 1).all(), \
        "Margin should be in [0, 1]"
    assert (uncertainty_df['max_prob'] >= 0).all() and (uncertainty_df['max_prob'] <= 1).all(), \
        "Max probability should be in [0, 1]"
    print("  ✅ All uncertainty signals in valid ranges")
    
    assert np.allclose(uncertainty_df['margin'], uncertainty_df['confidence_gap']), \
        "Margin and confidence_gap should be identical"
    print("  ✅ Margin and confidence_gap are aliased correctly")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: compute_uncertainty_signals() Working Correctly!")
    print("=" * 80 + "\n")


def test_edge_cases_uncertainty():
    """Test edge cases for uncertainty signals."""
    print("\n" + "=" * 80)
    print("TEST: Edge Cases for Uncertainty Signals")
    print("=" * 80)
    
    n_classes = 3
    
    # Edge case 1: Perfect certainty (one class = 1.0)
    print("\n1️⃣  Perfect certainty:")
    y_prob_certain = np.zeros((10, n_classes))
    y_prob_certain[:, 0] = 1.0
    
    unc_certain = uncertainty.compute_uncertainty_signals(y_prob_certain)
    print(f"  Entropy (should be ~0): {unc_certain['entropy'].mean():.6f}")
    print(f"  Margin (should be ~1.0): {unc_certain['margin'].mean():.6f}")
    print(f"  Max prob (should be 1.0): {unc_certain['max_prob'].mean():.6f}")
    assert unc_certain['entropy'].mean() < 0.01, "Entropy should be near 0"
    assert unc_certain['margin'].mean() > 0.99, "Margin should be near 1.0"
    print("  ✅ Perfect certainty handled correctly")
    
    # Edge case 2: Perfect uncertainty (uniform distribution)
    print("\n2️⃣  Perfect uncertainty (uniform):")
    y_prob_uniform = np.ones((10, n_classes)) / n_classes
    
    unc_uniform = uncertainty.compute_uncertainty_signals(y_prob_uniform)
    print(f"  Entropy (should be ~{np.log(n_classes):.4f}): {unc_uniform['entropy'].mean():.6f}")
    print(f"  Margin (should be 0): {unc_uniform['margin'].mean():.6f}")
    print(f"  Max prob (should be {1/n_classes:.4f}): {unc_uniform['max_prob'].mean():.6f}")
    assert unc_uniform['margin'].mean() < 0.01, "Margin should be near 0"
    print("  ✅ Perfect uncertainty handled correctly")
    
    print("\n" + "=" * 80)
    print("✅ EDGE CASE TESTS PASSED!")
    print("=" * 80 + "\n")


def test_classify_uncertainty_level():
    """Test uncertainty level classification."""
    print("\n" + "=" * 80)
    print("TEST: classify_uncertainty_level()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    uncertainty_df = uncertainty.compute_uncertainty_signals(y_prob)
    
    # Classify uncertainty levels
    classified_df = uncertainty.classify_uncertainty_level(uncertainty_df)
    
    print("\n📊 UNCERTAINTY LEVEL DISTRIBUTION")
    print("-" * 80)
    level_counts = classified_df['uncertainty_level'].value_counts()
    for level, count in level_counts.items():
        pct = 100 * count / len(classified_df)
        print(f"  {level:.<15} {count:>3} samples ({pct:>5.1f}%)")
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert 'uncertainty_level' in classified_df.columns, "Should add uncertainty_level column"
    assert set(classified_df['uncertainty_level'].unique()).issubset({'LOW', 'MEDIUM', 'HIGH'}), \
        "Should only have LOW, MEDIUM, HIGH levels"
    print("  ✅ Uncertainty classification working correctly")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: classify_uncertainty_level() Working Correctly!")
    print("=" * 80 + "\n")


def test_uncertainty_summary_stats():
    """Test uncertainty summary statistics."""
    print("\n" + "=" * 80)
    print("TEST: uncertainty_summary_stats()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    uncertainty_df = uncertainty.compute_uncertainty_signals(y_prob)
    
    # Get summary stats
    summary = uncertainty.uncertainty_summary_stats(uncertainty_df)
    
    print("\n📊 UNCERTAINTY SUMMARY STATISTICS")
    print("-" * 80)
    
    for signal_name, stats in summary.items():
        print(f"\n{signal_name.upper()}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name:.<15} {value:.4f}")
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(summary, dict), "Should return dict"
    assert set(summary.keys()) == {'entropy', 'margin', 'max_prob'}, "Should have all signal names"
    print("  ✅ Summary statistics computed correctly")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: uncertainty_summary_stats() Working Correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_compute_uncertainty_signals()
    test_edge_cases_uncertainty()
    test_classify_uncertainty_level()
    test_uncertainty_summary_stats()
    print("\n🎉 ALL UNCERTAINTY TESTS COMPLETED SUCCESSFULLY! 🎉\n")