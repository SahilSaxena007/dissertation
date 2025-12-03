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
    # You can tweak the budget here (e.g. 0.3 = 30% of cases escalated)
    review_budget = 0.9
    best = optimize_thresholds(review_budget=review_budget, num_thresholds=201)

    print("\n✅ Step 3 completed.")
    print(
        f"   Chosen τ* = {best['best_threshold']:.3f} "
        f"with review_rate = {best['review_rate']:.3f}, "
        f"policy_accuracy = {best['policy_accuracy']:.3f}, "
        f"AI-only = {best['ai_only_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
