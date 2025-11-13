"""
Components 3️⃣ 4️⃣ 5️⃣: Confusion matrix, ROC curves, calibration curves.
"""

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot normalized confusion matrix with counts and percentages.
    
    Parameters:
    - y_true, y_pred: True and predicted labels
    - class_names: List of class names
    - save_path: Where to save PNG
    """
    # TODO: Implement
    pass


def plot_roc_ovr(y_true, y_prob, class_names, save_path):
    """
    Plot One-vs-Rest ROC curves and macro-average AUC.
    
    Parameters:
    - y_true: True labels (integer encoded)
    - y_prob: Predicted probabilities (n_samples x n_classes)
    - class_names: List of class names
    - save_path: Where to save PNG
    """
    # TODO: Implement
    pass


def plot_calibration_curve(y_true, y_prob, class_names, save_path):
    """
    Plot calibration curves (predicted vs observed probability) per class.
    
    Parameters:
    - y_true, y_prob: True labels and predicted probabilities
    - class_names: List of class names
    - save_path: Where to save PNG
    """
    # TODO: Implement
    pass