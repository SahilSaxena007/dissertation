"""
Run Step 2 escalation on the whole ADNI test set.

Usage (from src/):
    python .\escalation\run_inference_batch.py
"""

import os
import sys
import joblib
import numpy as np

# --- make sure create_model exists in __main__ for joblib -------------------
from model_stub import create_model as _create_model  # noqa: F401
import __main__
__main__.create_model = _create_model  # attach for unpickling

from escalation_engine import run_batch_escalation, run_single_escalation
from escalation_config import CLASS_NAMES, ESCALATION_CONFIG


def main():
    artifacts_dir = os.path.join("..", "artifacts")

    print("📦 Loading ensemble models from voting_ensemble.pkl …")
    models = joblib.load(os.path.join(artifacts_dir, "voting_ensemble.pkl"))

    print("📂 Loading X_test / y_test …")
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")

    print("\n🚦 Running Step 2 escalation on test set …")
    df = run_batch_escalation(
        X=X_test,
        y_true=y_test,
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
    )

    out_dir = os.path.join("..", "reports", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "escalation_table_testset.csv")

    df.to_csv(out_path)
    print(f"\n✅ Escalation table saved to: {out_path}")
    print("\n🔎 Preview:")
    print(df.head(10))
    print("\n📊 Escalation counts:")
    print(df["escalation_level"].value_counts())
    
    


    # ------------------------------------------------------------------
    # DEBUG CHECK: compare batch vs single for sample_id = 0
    # ------------------------------------------------------------------
    print("\n🔁 Debug: comparing batch vs single for sample_id = 0 …")
    sample_id = 0
    x0 = X_test[sample_id]
    y0 = int(y_test[sample_id])
    
    print("\nDEBUG — RAW FEATURES COMPARISON")
    print("Single x[0]:", x0)
    print("Batch  x[0]:", X_test[0])
    print("Equal? ->", np.allclose(x0, X_test[0]))

    single_res = run_single_escalation(
        x_row=x0,
        models=models,
        class_names=CLASS_NAMES,
        config=ESCALATION_CONFIG,
        y_true=y0,
    )

    batch_row = df.loc[sample_id].to_dict()

    print("\n  Single       ens_max_prob:", single_res["ens_max_prob"])
    print("  Batch  [0]   ens_max_prob:", batch_row["ens_max_prob"])

    print("  Single       pred_ens    :", single_res["pred_ens"], single_res["pred_ens_name"])
    print("  Batch  [0]   pred_ens    :", batch_row["pred_ens"], batch_row["pred_ens_name"])

    print("  Single       escalation  :", single_res["escalation_level"])
    print("  Batch  [0]   escalation  :", batch_row["escalation_level"])


if __name__ == "__main__":
    main()
