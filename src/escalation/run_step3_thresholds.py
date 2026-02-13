"""
Runner for Step 3 - Threshold Optimisation

Usage (from src/):
    python .\\\\escalation\\\\run_step3_thresholds.py
"""

from threshold_optimizer import optimize_thresholds


def main():
    review_budget = 0.2
    human_accuracy = 0.95

    best = optimize_thresholds(
        review_budget=review_budget,
        human_accuracy=human_accuracy,
        num_thresholds=201,
    )

    print("\nStep 3 completed.")
    print(
        f"   Chosen tau* = {best['best_threshold']:.3f} "
        f"with review_rate = {best['review_rate']:.3f}, "
        f"policy_accuracy = {best['policy_accuracy']:.3f}, "
        f"AI-only = {best['ai_only_accuracy']:.3f}, "
        f"(human_accuracy_assumed = {best['human_accuracy_assumed']:.3f})"
    )


if __name__ == "__main__":
    main()
