"""
Test Components 3️⃣ 4️⃣ 5️⃣: Confusion matrix, ROC curves, calibration curves.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import visualizations
from utils import constants


def test_plot_confusion_matrix():
    """Test confusion matrix visualization."""
    print("\n" + "=" * 80)
    print("TEST: plot_confusion_matrix()")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=int(0.2 * len(y_true)), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, size=len(error_indices))
    
    # Create output directory if it doesn't exist
    os.makedirs("./reports/figures", exist_ok=True)
    
    # Plot
    save_path = "./reports/figures/test_confusion_matrix.png"
    fig = visualizations.plot_confusion_matrix(y_true, y_pred, constants.CLASS_NAMES, save_path)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None, "Figure object should be returned"
    print(f"\n✅ Confusion matrix saved successfully!")
    
    print("=" * 80 + "\n")


def test_plot_roc_ovr():
    """Test ROC curves visualization."""
    print("\n" + "=" * 80)
    print("TEST: plot_roc_ovr()")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    # Create probabilities with some structure
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    os.makedirs("./reports/figures", exist_ok=True)
    
    # Plot
    save_path = "./reports/figures/test_roc_curves.png"
    fig = visualizations.plot_roc_ovr(y_true, y_prob, constants.CLASS_NAMES, save_path)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None, "Figure object should be returned"
    print(f"\n✅ ROC curves saved successfully!")
    
    print("=" * 80 + "\n")


def test_plot_calibration_curve():
    """Test calibration curves visualization."""
    print("\n" + "=" * 80)
    print("TEST: plot_calibration_curve()")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(constants.RANDOM_SEED)
    n_samples = 300
    n_classes = 3
    
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(y_true)
    
    # Create probabilities with some structure
    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i, y_true[i]] = np.random.uniform(0.5, 1.0)
        remaining_prob = 1.0 - y_prob[i, y_true[i]]
        other_indices = np.delete(np.arange(n_classes), y_true[i])
        y_prob[i, other_indices] = np.random.dirichlet(np.ones(n_classes - 1)) * remaining_prob
    
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    os.makedirs("./reports/figures", exist_ok=True)
    
    # Plot
    save_path = "./reports/figures/test_calibration_curves.png"
    fig = visualizations.plot_calibration_curve(y_true, y_prob, constants.CLASS_NAMES, save_path)
    
    assert os.path.exists(save_path), f"Figure not saved to {save_path}"
    assert fig is not None, "Figure object should be returned"
    print(f"\n✅ Calibration curves saved successfully!")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_plot_confusion_matrix()
    test_plot_roc_ovr()
    test_plot_calibration_curve()
    print("\n🎉 ALL VISUALIZATION TESTS COMPLETED SUCCESSFULLY! 🎉\n")