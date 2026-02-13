"""
Runner for Step 3 - Threshold Optimisation

Usage (from src/):
    python .\\\\escalation\\\\run_step3_thresholds.py
"""

import numpy as np

from threshold_optimizer import optimize_thresholds_with_sensitivity


def main():
    review_budget = 0.2
    # Literature-guided clinician performance range for sensitivity analysis.
    human_accuracy_values = [round(v, 2) for v in np.arange(0.85, 0.951, 0.01)]

    # Keep fixed-mode function available, but default to sensitivity-aware mode.
    best = optimize_thresholds_with_sensitivity(
        review_budget=review_budget,
        human_accuracy_values=human_accuracy_values,
        num_thresholds=201,
    )

    print("\nStep 3 completed.")
    print(
        f"   Chosen tau* = {best['best_threshold']:.3f} "
        f"with review_rate = {best['review_rate']:.3f}, "
        f"policy_accuracy = {best['policy_accuracy']:.3f}, "
        f"AI-only = {best['ai_only_accuracy']:.3f}, "
        f"(human_accuracy_range = {best['human_accuracy_range'][0]:.2f}-"
        f"{best['human_accuracy_range'][1]:.2f})"
    )


if __name__ == "__main__":
    main()
