"""
Test Component 1️⃣ 2️⃣: Overall and per-class metrics.
Tests the updated metrics.py implementation.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import metrics
from utils import constants


def test_compute_classification_metrics():
    """Test overall and per-class metrics computation."""
    print("\n" + "=" * 80)
    print("TEST: compute_classification_metrics()")
    print("=" * 80)
    
    # Create synthetic test data (balanced)
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    # True labels (balanced)
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    # Predicted labels (some errors)
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
    
    # Predicted probabilities (mostly correct, with some noise)
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        # High probability on true class
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        # Low probabilities on others
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    # Normalize to ensure sum = 1
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Compute metrics
    overall_metrics, per_class_df = metrics.compute_classification_metrics(
        y_true, y_pred, y_prob, constants.CLASS_NAMES
    )
    
    # ─────────────────────────────────────────
    # 📊 OVERALL METRICS (Component 1️⃣)
    # ─────────────────────────────────────────
    print("\n📊 OVERALL METRICS (Component 1️⃣)")
    print("-" * 80)
    
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            status = "✅" if not np.isnan(value) else "⚠️ "
            print(f"  {status} {key:.<35} {value:.4f}")
        else:
            print(f"  ✅ {key:.<35} {value}")
    
    # ─────────────────────────────────────────
    # 📊 PER-CLASS METRICS (Component 2️⃣)
    # ─────────────────────────────────────────
    print("\n📊 PER-CLASS METRICS (Component 2️⃣)")
    print("-" * 80)
    print(per_class_df.to_string(index=False))
    
    # ─────────────────────────────────────────
    # ✅ ASSERTIONS
    # ─────────────────────────────────────────
    print("\n✅ ASSERTIONS")
    print("-" * 80)
    
    # Check shapes
    assert isinstance(overall_metrics, dict), "overall_metrics should be dict"
    assert isinstance(per_class_df, pd.DataFrame), "per_class_df should be DataFrame"
    assert per_class_df.shape[0] == n_classes, f"per_class_df should have {n_classes} rows"
    assert per_class_df.shape[1] == 5, "per_class_df should have 5 columns (Class, Precision, Recall, F1, AUC)"
    print("  ✅ Return types and shapes correct")
    
    # Check metric ranges
    assert 0 <= overall_metrics["accuracy"] <= 1, "accuracy should be in [0, 1]"
    assert 0 <= overall_metrics["macro_precision"] <= 1, "macro_precision should be in [0, 1]"
    assert 0 <= overall_metrics["macro_recall"] <= 1, "macro_recall should be in [0, 1]"
    assert 0 <= overall_metrics["macro_f1"] <= 1, "macro_f1 should be in [0, 1]"
    print("  ✅ Overall metrics in valid ranges [0, 1]")
    
    # Check per-class metrics
    assert (per_class_df["Precision"] >= 0).all() and (per_class_df["Precision"] <= 1).all(), \
        "Precision should be in [0, 1]"
    assert (per_class_df["Recall"] >= 0).all() and (per_class_df["Recall"] <= 1).all(), \
        "Recall should be in [0, 1]"
    assert (per_class_df["F1"] >= 0).all() and (per_class_df["F1"] <= 1).all(), \
        "F1 should be in [0, 1]"
    assert (per_class_df["AUC"] >= 0).all() and (per_class_df["AUC"] <= 1).all(), \
        "AUC should be in [0, 1]"
    print("  ✅ Per-class metrics in valid ranges [0, 1]")
    
    # Check class names
    assert list(per_class_df["Class"]) == constants.CLASS_NAMES, \
        f"Class names mismatch. Expected {constants.CLASS_NAMES}, got {list(per_class_df['Class'])}"
    print("  ✅ Class names match constants")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED: Components 1️⃣ 2️⃣ Working Correctly!")
    print("=" * 80 + "\n")


def test_per_class_metrics_table():
    """Test per-class metrics computation (standalone)."""
    print("\n" + "=" * 80)
    print("TEST: per_class_metrics_table()")
    print("=" * 80)
    
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 100
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    per_class_df = metrics.per_class_metrics_table(y_true, y_pred, y_prob, constants.CLASS_NAMES)
    
    print("\nPer-class metrics table:")
    print("-" * 80)
    print(per_class_df.to_string(index=False))
    
    assert per_class_df.shape[0] == n_classes, f"Should have {n_classes} rows"
    assert set(per_class_df.columns) == {"Class", "Precision", "Recall", "F1", "AUC"}, \
        f"Column names mismatch. Got {set(per_class_df.columns)}"
    
    print("\n✅ per_class_metrics_table() test passed!")
    print("=" * 80 + "\n")


# ... existing code ...

def test_edge_cases():
    """Test edge cases (perfect predictions, all same class, etc.)."""
    print("\n" + "=" * 80)
    print("TEST: Edge Cases")
    print("=" * 80)

    n_samples = 99  # ← CHANGE: Use multiple of 3 for balanced split
    n_classes = 3

    # Edge case 1: Perfect predictions
    print("\n1️⃣  Perfect predictions:")
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    y_pred = y_true.copy()
    y_prob = np.zeros((len(y_true), n_classes))  # ← Use len(y_true) instead of n_samples
    y_prob[np.arange(len(y_true)), y_true] = 1.0

    overall_metrics, per_class_df = metrics.compute_classification_metrics(
        y_true, y_pred, y_prob, constants.CLASS_NAMES
    )
    assert overall_metrics["accuracy"] == 1.0, "Perfect predictions should have accuracy=1.0"
    assert overall_metrics["macro_f1"] == 1.0, "Perfect predictions should have F1=1.0"
    print("  ✅ Perfect predictions: accuracy=1.0, F1=1.0")

    # Edge case 2: All predictions same class (should not crash)
    print("\n2️⃣  All predictions same class:")
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.zeros(n_samples, dtype=int)  # All predict class 0
    y_prob = np.zeros((n_samples, n_classes))
    y_prob[:, 0] = 1.0

    overall_metrics, per_class_df = metrics.compute_classification_metrics(
        y_true, y_pred, y_prob, constants.CLASS_NAMES
    )
    print("  ✅ Handled all-same-class predictions without crashing")
    print(f"     Accuracy: {overall_metrics['accuracy']:.4f}")

    # Edge case 3: Single sample per class
    print("\n3️⃣  Single sample per class:")
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_prob = np.eye(n_classes)

    overall_metrics, per_class_df = metrics.compute_classification_metrics(
        y_true, y_pred, y_prob, constants.CLASS_NAMES
    )
    print("  ✅ Handled single sample per class without crashing")
    print(f"     Accuracy: {overall_metrics['accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("✅ ALL EDGE CASE TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_compute_classification_metrics()
    test_per_class_metrics_table()
    test_edge_cases()
    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉\n")