"""
Run Step 2 escalation for a single patient (from the test set).

Usage (from src/):
    python .\escalation\run_inference_single.py          # uses sample_id = 0
    python .\escalation\run_inference_single.py 42       # uses sample_id = 42
"""

import sys
import os
import joblib
import numpy as np
from pprint import pprint

# --- stub so joblib can unpickle the SciKeras model ------------------------
from model_stub import create_model as _create_model  # noqa: F401
import __main__
__main__.create_model = _create_model  # attach for unpickling

from engine import run_single_escalation
from config import CLASS_NAMES, ESCALATION_CONFIG


def main():
    # choose index from CLI or default 0
    if len(sys.argv) > 1:
        sample_id = int(sys.argv[1])
    else:
        sample_id = 0

    artifacts_dir = os.path.join("..", "artifacts")

    print("Loading ensemble models...")
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))

    print("Loading X_test / y_test...")
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    if sample_id < 0 or sample_id >= len(X_test):
        raise ValueError(f"sample_id {sample_id} out of range (0..{len(X_test)-1})")

    x = X_test[sample_id]
    y = int(y_test[sample_id])

    print(f"\nRunning Step 2 for sample_id={sample_id} (true label = {CLASS_NAMES[y]})...")
    
    print("Loaded x:", x)

    result = run_single_escalation(
        x_row=x,
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
        y_true=y,
    )

    print("\nResult:")
    pprint(result)

    print("\nInterpretation:")
    print(f"  • Model prediction       : {result['pred_ens_name']} (index {result['pred_ens']})")
    print(f"  • Escalation level       : {result['escalation_level']}")
    print(f"  • Reasons                : {result['escalation_reasons'] or 'none (AI-Autonomous)'}")
    print(f"  • Max prob / margin      : {result['ens_max_prob']:.3f} / {result['ens_margin']:.3f}")
    print(f"  • Entropy                : {result['ens_entropy']:.3f}")
    print(
        f"  • Model disagreement?    : {result['model_disagreement']} "
        f"(score={result['disagreement_score']:.3f})"
    )
    print(f"  • Missing fraction       : {result['missing_fraction']:.3f}")


if __name__ == "__main__":
    main()
