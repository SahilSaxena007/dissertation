"""
Component 🔟: Subgroup performance analysis (by sex, age, scanner, etc.).
"""

def evaluate_by_subgroup(y_true, y_pred, y_prob, class_names, df_metadata, save_path=None):
    """
    Evaluate model performance stratified by demographic/technical subgroups.
    
    Parameters:
    - y_true, y_pred, y_prob: Predictions
    - class_names: List of class names
    - df_metadata: DataFrame with columns like 'PTGENDER', 'AGE', 'SCANNER'
    - save_path: Optional path to save results
    
    Returns:
    - subgroup_results: Dict mapping subgroup name to metrics
    """
    # TODO: Implement
    pass