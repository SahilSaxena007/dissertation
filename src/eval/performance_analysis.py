"""
🎯 ORCHESTRATOR: Master function that chains all 11 components.
"""

# Import sibling modules directly (not via eval package)
from . import metrics
from . import visualizations
from . import uncertainty
from . import statistical_inference
from . import error_taxonomy
from . import explainability
from . import bias_diagnostics
from . import reporting


def analyze_model_performance(y_true, y_pred, y_prob, model_name, class_names, output_dir="../reports"):
    """
    Master function: orchestrates metric computation, CI estimation, visualization, 
    and file saving for all 11 components.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities
    - model_name: Name of the model (for saving files)
    - class_names: List of class names (e.g., ["SCD", "MCI", "AD"])
    - output_dir: Directory to save reports and plots (default: ../reports)
    
    Returns:
    - summary_dict: Compact dictionary with key metrics for comparison
    """
    # TODO: Chain all 11 components
    pass