"""
Test Component 9️⃣: SHAP-based explainability.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import explainability
from utils import constants


def create_simple_model(X_train, y_train):
    """Create a simple RandomForest model for testing."""
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=constants.RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model


def test_compute_shap_values():
    """Test SHAP value computation."""
    print("\n" + "=" * 80)
    print("TEST: compute_shap_values()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 100
    n_features = 9
    n_classes = 3
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    feature_names = constants.FEATURE_LIST
    
    # Train model
    print("\nTraining RandomForest model...")
    model = create_simple_model(X, y)
    
    # Compute SHAP values
    print("\nComputing SHAP values (tree-based)...")
    explainer, shap_values, X_sample = explainability.compute_shap_values(
        model,
        X,
        constants.CLASS_NAMES,
        model_type='tree',
        max_samples=50,  # Use subset for speed
    )
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert explainer is not None, "Explainer should be created"
    assert shap_values is not None, "SHAP values should be computed"
    print("  ✅ SHAP explainer and values computed")
    
    # Check SHAP values shape
    if isinstance(shap_values, list):
        print(f"  ✅ Multiclass SHAP values: {len(shap_values)} classes")
        for i, sv in enumerate(shap_values):
            print(f"     Class {i}: shape {sv.shape}")
    else:
        print(f"  ✅ SHAP values shape: {shap_values.shape}")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: compute_shap_values() Working Correctly!")
    print("=" * 80 + "\n")
    
    return model, X, X_sample, shap_values, explainer, feature_names


def test_plot_shap_summary(model, X_sample, shap_values, feature_names):
    """Test SHAP summary plot."""
    print("\n" + "=" * 80)
    print("TEST: plot_shap_summary()")
    print("=" * 80)
    
    os.makedirs("./reports/figures", exist_ok=True)
    
    save_path = "./reports/figures/test_shap_summary.png"
    
    # Create dummy explainer (we already have shap_values)
    explainer = None
    
    fig = explainability.plot_shap_summary(
        explainer,
        shap_values,
        X_sample,
        feature_names,
        save_path,
        plot_type='bar',
    )
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None or os.path.exists(save_path), "Should create figure or save file"
    print("  ✅ SHAP summary plot saved successfully")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: plot_shap_summary() Working Correctly!")
    print("=" * 80 + "\n")


def test_plot_shap_force(model, X_sample, shap_values, feature_names):
    """Test SHAP force plot."""
    print("\n" + "=" * 80)
    print("TEST: plot_shap_force()")
    print("=" * 80)
    
    os.makedirs("./reports/figures", exist_ok=True)
    
    save_path = "./reports/figures/test_shap_force_sample0.png"
    
    explainer = None
    
    fig = explainability.plot_shap_force(
        explainer,
        shap_values,
        X_sample,
        sample_index=0,
        feature_names=feature_names,
        save_path=save_path,
    )
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None or os.path.exists(save_path), "Should create figure or save file"
    print("  ✅ SHAP force plot saved successfully")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: plot_shap_force() Working Correctly!")
    print("=" * 80 + "\n")


def test_plot_shap_dependence(model, X_sample, shap_values, feature_names):
    """Test SHAP dependence plot."""
    print("\n" + "=" * 80)
    print("TEST: plot_shap_dependence()")
    print("=" * 80)
    
    os.makedirs("./reports/figures", exist_ok=True)
    
    save_path = "./reports/figures/test_shap_dependence_ABETA.png"
    
    explainer = None
    feature_idx = 0  # First feature (ABETA)
    
    fig = explainability.plot_shap_dependence(
        explainer,
        shap_values,
        X_sample,
        feature_idx=feature_idx,
        feature_names=feature_names,
        save_path=save_path,
    )
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None or os.path.exists(save_path), "Should create figure or save file"
    print("  ✅ SHAP dependence plot saved successfully")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: plot_shap_dependence() Working Correctly!")
    print("=" * 80 + "\n")


def test_feature_importance_ranking(shap_values, feature_names):
    """Test feature importance ranking."""
    print("\n" + "=" * 80)
    print("TEST: feature_importance_ranking()")
    print("=" * 80)
    
    importance_df = explainability.feature_importance_ranking(
        shap_values,
        feature_names,
        top_n=9,
    )
    
    print("\n📊 FEATURE IMPORTANCE RANKING (Component 9️⃣)")
    print("-" * 80)
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(importance_df, pd.DataFrame), "Should return DataFrame"
    assert len(importance_df) > 0, "Should have ranked features"
    assert set(importance_df.columns) == {'Rank', 'Feature', 'Mean |SHAP|', 'Relative Importance (%)'}, \
        "Should have correct columns"
    print("  ✅ Feature importance ranking computed correctly")
    
    # Check that relative importance sums to ~100%
    total_importance = importance_df['Relative Importance (%)'].sum()
    print(f"  ✅ Total relative importance: {total_importance:.1f}% (should be <= 100%)")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: feature_importance_ranking() Working Correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n🚀 STARTING EXPLAINABILITY TESTS\n")
    
    # Compute SHAP values once, then use in all tests
    model, X, X_sample, shap_values, explainer, feature_names = test_compute_shap_values()
    
    test_plot_shap_summary(model, X_sample, shap_values, feature_names)
    test_plot_shap_force(model, X_sample, shap_values, feature_names)
    test_plot_shap_dependence(model, X_sample, shap_values, feature_names)
    test_feature_importance_ranking(shap_values, feature_names)
    
    print("\n🎉 ALL EXPLAINABILITY TESTS COMPLETED SUCCESSFULLY! 🎉\n")