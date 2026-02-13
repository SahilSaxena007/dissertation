# dissertation

Human-in-the-loop (HITL) Alzheimer's classification pipeline using ADNI biomarkers. The system predicts `SCD` / `MCI` / `AD` with an ensemble (CatBoost, Random Forest, Neural Network), then applies escalation logic for clinician review.

## Repository Layout
- `src/`: runnable pipeline scripts
- `Study_files/`: raw ADNI exports
- `data/`: merged and preprocessed CSVs
- `artifacts/`: models, preprocessors, OOF/test arrays
- `Outputs/`, `reports/`: figures/tables
- `tests/`: standalone validation scripts

## Environment Setup
Run from `src/`.

```powershell
cd src
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## End-to-End Pipeline (OOF-first)
Run in this order from `src/`:

```powershell
python merge_files.py
python EDA.py
python Preprocess.py
python KNN_test.py
python selectkbest_test.py
python ModelsFinal.py
python run_evaluation.py
python .\escalation\run_inference_batch.py --source oof
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
```

### Step outputs
1. `merge_files.py` -> `../data/merged_data.csv`
2. `Preprocess.py` -> `../data/preprocessed_data.csv`
3. `ModelsFinal.py` ->
   - Repeated stratified CV fold metrics: `../Outputs/cv_fold_metrics.csv`
   - CV summary metrics (mean/std): `../Outputs/overall_metrics.csv`
   - OOF probabilities/labels (`../artifacts/oof_*.npy`) generated on the train split only
   - Strict locked holdout artifacts for unseen demo:
     - `../artifacts/X_test.npy`, `../artifacts/y_test.npy` (20% untouched holdout)
     - `../artifacts/X_train.npy`, `../artifacts/y_train.npy`
     - `../artifacts/train_indices.npy`, `../artifacts/holdout_indices.npy`
   - Final model artifacts fit on train split only (not full dataset)
   - Calibration artifacts:
     - `../artifacts/oof_ens_proba_calibrated_sigmoid.npy`
     - `../artifacts/oof_ens_proba_calibrated_isotonic.npy`
     - `../artifacts/oof_ens_proba_selected.npy` (auto-selected from raw/sigmoid/isotonic by OOF CV log-loss, tie-break by ECE)
     - `../artifacts/oof_ens_proba_calibrated.npy` (compatibility alias to selected artifact)
     - `../artifacts/calibration_summary.json`
4. `run_evaluation.py` -> uses OOF artifacts by default for leak-free evaluation
5. `run_inference_batch.py --source oof` -> `../reports/tables/escalation_table_oof.csv` (uses selected ensemble probabilities for uncertainty signals when available)
6. `step2_feature_builder.py` -> `../reports/tables/step2_meta_dataset.csv`
7. `step2_train_meta_model.py` ->
   - CV out-of-sample risk scores in `../reports/tables/threshold_data.csv`
   - Stratified bootstrap evaluation summary in `../reports/tables/step2_meta_bootstrap_metrics.csv`
8. `run_step3_thresholds.py` ->
   - `../reports/tables/threshold_analysis.csv` (expected policy metrics under human-accuracy sensitivity)
   - `../reports/tables/threshold_analysis_by_human_accuracy.csv` (full per-threshold, per-scenario sweep)
   - `../reports/tables/threshold_sensitivity_summary.csv` (best threshold per human-accuracy scenario)
   - `../artifacts/step3_best_threshold.json` (robust threshold chosen from sensitivity-aware expected accuracy)

## Unseen Holdout Mode
`python .\escalation\run_inference_batch.py --source testset` runs escalation on the strict untouched holdout set (`X_test.npy`, `y_test.npy`) for unseen-data demo.

## Tests
Run from repository root:

```powershell
python tests/test_imports.py
python tests/test_metrics.py
python tests/test_uncertainty.py
python tests/test_statistical.py
python tests/test_explainability.py
python tests/test_taxonomy.py
python tests/test_bias.py
python tests/test_visualization.py
```

## Notes
- Scripts use relative paths and expect CWD=`src/`.
- Class mapping is fixed: `SCD=0`, `MCI=1`, `AD=2`.
- Keep docs synced after pipeline changes: `README.md`, `AGENTS.md`, `CLAUDE.md`.
