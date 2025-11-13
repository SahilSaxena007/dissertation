"""
Test Component 7️⃣: Error taxonomy export and analysis.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import error_taxonomy
from utils import constants


def test_export_error_taxonomy():
    """Test error taxonomy export."""
    print("\n" + "=" * 80)
    print("TEST: export_error_taxonomy()")
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
    
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Create sample IDs
    sample_ids = np.arange(1000, 1000 + n_samples)
    
    # Create optional metadata
    metadata_df = pd.DataFrame({
        'age': np.random.uniform(50, 90, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'scanner': np.random.choice(['Siemens', 'GE', 'Philips'], n_samples),
    })
    
    # Create output directory
    os.makedirs("./reports/tables", exist_ok=True)
    
    # Export error taxonomy
    save_path = "./reports/tables/test_error_taxonomy_catboost.csv"
    error_df = error_taxonomy.export_error_taxonomy(
        y_true,
        y_pred,
        y_prob,
        constants.CLASS_NAMES,
        model_name="CatBoost",
        save_path=save_path,
        sample_ids=sample_ids,
        metadata_df=metadata_df,
    )
    
    print("\n📊 ERROR TAXONOMY DATAFRAME (first 10 rows)")
    print("-" * 80)
    print(error_df.head(10)[['sample_id', 'true_label', 'predicted_label', 'is_correct', 'entropy', 'margin']].to_string(index=False))
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert os.path.exists(save_path), f"File not saved to {save_path}"
    assert isinstance(error_df, pd.DataFrame), "Should return DataFrame"
    assert len(error_df) == n_samples, f"Should have {n_samples} rows"
    print("  ✅ File saved successfully")
    print(f"  ✅ DataFrame has correct shape: {error_df.shape}")
    
    # Check columns
    expected_cols = {
        'sample_id', 'sample_index', 'true_label', 'predicted_label',
        'entropy', 'margin', 'max_prob', 'is_correct', 'is_error',
        'model_name', 'timestamp'
    }
    assert expected_cols.issubset(error_df.columns), "Should have all expected columns"
    print("  ✅ All expected columns present")
    
    # Check probability columns
    prob_cols = [f'prob_{cn}' for cn in constants.CLASS_NAMES]
    assert all(col in error_df.columns for col in prob_cols), "Should have probability columns"
    print("  ✅ Probability columns present")
    
    # Check metadata columns
    assert 'age' in error_df.columns and 'sex' in error_df.columns, \
        "Should have metadata columns"
    print("  ✅ Metadata columns included")
    
    # Check correctness
    n_correct = error_df['is_correct'].sum()
    n_total = len(error_df)
    accuracy = n_correct / n_total
    print(f"  ✅ Accuracy from error taxonomy: {accuracy:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: export_error_taxonomy() Working Correctly!")
    print("=" * 80 + "\n")
    
    return error_df


def test_error_taxonomy_summary(error_df):
    """Test error taxonomy summary statistics."""
    print("\n" + "=" * 80)
    print("TEST: error_taxonomy_summary()")
    print("=" * 80)
    
    summary = error_taxonomy.error_taxonomy_summary(error_df, constants.CLASS_NAMES)
    
    print("\n📊 ERROR TAXONOMY SUMMARY")
    print("-" * 80)
    print(f"  Total samples:.......... {summary['total_samples']}")
    print(f"  Correct predictions:... {summary['correct_predictions']}")
    print(f"  Error predictions:..... {summary['error_predictions']}")
    print(f"  Accuracy:.............. {summary['accuracy']:.4f}")
    print(f"  Error rate:............ {summary['error_rate']:.4f}")
    
    print("\n📊 UNCERTAINTY STATS (Correct vs. Error)")
    print("-" * 80)
    print(f"  Correct predictions:")
    print(f"    Entropy:............. {summary['uncertainty_stats']['correct']['entropy']:.4f}")
    print(f"    Margin:............. {summary['uncertainty_stats']['correct']['margin']:.4f}")
    print(f"    Max prob:........... {summary['uncertainty_stats']['correct']['max_prob']:.4f}")
    
    print(f"\n  Error predictions:")
    print(f"    Entropy:............. {summary['uncertainty_stats']['error']['entropy']:.4f}")
    print(f"    Margin:............. {summary['uncertainty_stats']['error']['margin']:.4f}")
    print(f"    Max prob:........... {summary['uncertainty_stats']['error']['max_prob']:.4f}")
    
    print("\n📊 CONFUSION MATRIX")
    print("-" * 80)
    for key, value in sorted(summary['confusion_matrix'].items()):
        print(f"  {key:.<30} {value}")
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(summary, dict), "Should return dict"
    assert summary['accuracy'] + summary['error_rate'] == 1.0, \
        "Accuracy + error_rate should equal 1.0"
    print("  ✅ Summary statistics valid")
    
    # Errors should have higher entropy
    if summary['uncertainty_stats']['error']['entropy'] > 0:
        assert summary['uncertainty_stats']['error']['entropy'] > \
               summary['uncertainty_stats']['correct']['entropy'], \
            "Errors should have higher entropy"
        print("  ✅ Errors have higher entropy (more uncertain)")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: error_taxonomy_summary() Working Correctly!")
    print("=" * 80 + "\n")


def test_error_cases_by_uncertainty(error_df):
    """Test extraction of most uncertain error cases."""
    print("\n" + "=" * 80)
    print("TEST: error_cases_by_uncertainty()")
    print("=" * 80)
    
    n_top = 5
    top_uncertain = error_taxonomy.error_cases_by_uncertainty(error_df, n_top=n_top)
    
    print(f"\n📊 TOP {n_top} MOST UNCERTAIN ERROR CASES (For HITL Review)")
    print("-" * 80)
    print(top_uncertain[[
        'sample_id', 'true_label', 'predicted_label', 'entropy', 'margin', 'max_prob'
    ]].to_string(index=False))
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert isinstance(top_uncertain, pd.DataFrame), "Should return DataFrame"
    assert len(top_uncertain) <= n_top, f"Should have at most {n_top} rows"
    assert (top_uncertain['is_error'] == 1).all(), "Should only include errors"
    print("  ✅ Top uncertain cases correctly identified")
    
    # Check sorted by entropy (descending)
    entropies = top_uncertain['entropy'].values
    assert all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1)), \
        "Should be sorted by entropy (descending)"
    print("  ✅ Sorted correctly by entropy (descending)")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: error_cases_by_uncertainty() Working Correctly!")
    print("=" * 80 + "\n")


def test_export_meta_model_training_data():
    """Test meta-model training data export."""
    print("\n" + "=" * 80)
    print("TEST: export_meta_model_training_data()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 100
    n_classes = 3
    
    # Create 3 error taxonomies from 3 different models
    error_dfs = []
    for model_idx, model_name in enumerate(["CatBoost", "RandomForest", "NeuralNetwork"]):
        y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
        np.random.shuffle(y_true)
        
        y_pred = y_true.copy()
        error_indices = np.random.choice(len(y_true), size=int(0.15 * len(y_true)), replace=False)
        y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
        
        y_prob = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
            remaining_prob = 1.0 - y_prob[i, y_true[i]]
            other_indices = np.delete(np.arange(n_classes), y_true[i])
            y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
        
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Export error taxonomy for this model
        error_df = error_taxonomy.export_error_taxonomy(
            y_true, y_pred, y_prob,
            constants.CLASS_NAMES,
            model_name=model_name,
            save_path=f"./reports/tables/test_error_taxonomy_{model_name.lower()}.csv",
        )
        error_dfs.append(error_df)
    
    os.makedirs("./reports/tables", exist_ok=True)
    
    # Export combined meta-model training data
    save_path = "./reports/tables/test_meta_model_training_data.csv"
    meta_training_df = error_taxonomy.export_meta_model_training_data(
        error_dfs, constants.CLASS_NAMES, save_path
    )
    
    print("\n📊 META-MODEL TRAINING DATA (first 10 rows)")
    print("-" * 80)
    print(meta_training_df.head(10)[['sample_id', 'model_name', 'entropy', 'margin', 'is_correct']].to_string(index=False))
    
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    assert os.path.exists(save_path), "File not saved"
    assert isinstance(meta_training_df, pd.DataFrame), "Should return DataFrame"
    assert len(meta_training_df) == len(error_dfs) * n_samples, \
        "Should have combined samples from all models"
    print("  ✅ Meta-model training data saved successfully")
    print(f"  ✅ Total rows: {len(meta_training_df)} (3 models × {n_samples} samples)")
    
    # Check model distribution
    model_counts = meta_training_df['model_name'].value_counts()
    print(f"\n  Model distribution:")
    for model, count in model_counts.items():
        print(f"    {model}: {count}")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED: export_meta_model_training_data() Working Correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n🚀 STARTING ERROR TAXONOMY TESTS\n")
    
    error_df = test_export_error_taxonomy()
    test_error_taxonomy_summary(error_df)
    test_error_cases_by_uncertainty(error_df)
    test_export_meta_model_training_data()
    
    print("\n🎉 ALL ERROR TAXONOMY TESTS COMPLETED SUCCESSFULLY! 🎉\n")