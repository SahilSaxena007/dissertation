"""
Runner for Step 3 – Threshold Optimisation

Usage (from src/):
    python .\escalation\run_step3_thresholds.py

This will:
    - load threshold_data.csv
    - run the optimisation
    - print best τ under the default review budget
"""

from threshold_optimizer import optimize_thresholds


def main():
    # You can tweak these:
    review_budget = 0.2        # e.g. 0.2 = 20% cases escalated
    human_accuracy = 1     # assumed clinician accuracy on reviewed cases

    best = optimize_thresholds(
        review_budget=review_budget,
        human_accuracy=human_accuracy,
        num_thresholds=201,
    )

    print("\n✅ Step 3 completed.")
    print(
        f"   Chosen τ* = {best['best_threshold']:.3f} "
        f"with review_rate = {best['review_rate']:.3f}, "
        f"policy_accuracy = {best['policy_accuracy']:.3f}, "
        f"AI-only = {best['ai_only_accuracy']:.3f}, "
        f"(human_accuracy_assumed = {best['human_accuracy_assumed']:.3f})"
    )


if __name__ == "__main__":
    main()
