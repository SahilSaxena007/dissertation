# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Runner for Step 3 - Threshold Optimisation

Usage (from src/):
    python .\\\\escalation\\\\run_step3_thresholds.py
"""

import numpy as np

from threshold_optimizer import optimize_thresholds_with_sensitivity


def build_human_accuracy_grid(min_value: float = 0.85, max_value: float = 0.95, step: float = 0.01):
    if step <= 0:
        raise ValueError("step must be > 0.")
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value.")
    # Include max value when floating point error would otherwise drop it.
    values = np.arange(min_value, max_value + (step / 2.0), step)
    return [round(float(v), 2) for v in values]


def main():
    review_budget = 0.4
    # Literature-guided clinician performance range for sensitivity analysis.
    human_accuracy_values = build_human_accuracy_grid(min_value=0.85, max_value=0.95, step=0.01)

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
