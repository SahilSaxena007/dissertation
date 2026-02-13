"""
Runner for simulated-clinician HITL experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .feedback_loop import run_active_learning_simulation
except ImportError:
    from feedback_loop import run_active_learning_simulation


def run_experiments(
    clinician_accuracies: list[float],
    batch_size: int = 20,
    max_rounds: int = 6,
    use_confidence_weighting: bool = True,
    seed: int = 42,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for acc in clinician_accuracies:
        df = run_active_learning_simulation(
            clinician_accuracy=float(acc),
            batch_size=batch_size,
            max_rounds=max_rounds,
            use_confidence_weighting=use_confidence_weighting,
            seed=seed,
        )
        df["clinician_accuracy"] = float(acc)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "hitl_simulated_clinician_experiment.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accuracies",
        nargs="+",
        type=float,
        default=[0.85, 0.9, 0.95],
        help="Simulated clinician accuracies.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-confidence-weighting", action="store_true")
    args = parser.parse_args()

    out_path = run_experiments(
        clinician_accuracies=args.accuracies,
        batch_size=args.batch_size,
        max_rounds=args.max_rounds,
        use_confidence_weighting=not args.no_confidence_weighting,
        seed=args.seed,
    )
    print(f"Saved simulated clinician experiment to: {out_path}")


if __name__ == "__main__":
    main()
