# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions
When starting a new session, refer to `MASTERPLAN.md` for current phase context.

## Documentation Sync Rule
Whenever pipeline logic, artifact paths, or execution commands change, update all three files together in the same change:
- `CLAUDE.md`
- `AGENTS.md`
- `README.md`

## Project Overview
Dissertation project: **Human-in-the-Loop (HITL) framework for Alzheimer's disease classification** using ADNI biomarker data. The system classifies patients into **SCD/MCI/AD** and escalates uncertain/error-prone cases for clinician review.

## Pipeline Architecture (OOF-first)
All scripts assume `src/` as CWD and use relative paths (`../data`, `../artifacts`, `../reports`).

1. `merge_files.py` -> `data/merged_data.csv`
2. `Preprocess.py` -> `data/preprocessed_data.csv`
3. `ModelsFinal.py` ->
   - repeated stratified CV artifacts: `Outputs/cv_fold_metrics.csv` and `Outputs/overall_metrics.csv` (mean/std across folds)
   - OOF artifacts for leak-free evaluation: `oof_cat_proba.npy`, `oof_rf_proba.npy`, `oof_nn_proba.npy`, `oof_ens_proba.npy`, `X_oof.npy`, `y_oof.npy`
   - calibration artifacts: `oof_ens_proba_calibrated_sigmoid.npy`, `oof_ens_proba_calibrated_isotonic.npy`, `oof_ens_proba_selected.npy`, `oof_ens_proba_calibrated.npy` (compat alias), `calibration_summary.json`, calibrator pickles
   - deployment artifacts: models, imputer, selector, scaler
   - legacy split arrays for compatibility: `X_test.npy`, `y_test.npy`, etc.
4. `run_evaluation.py` -> evaluates with OOF artifacts by default (falls back to legacy test split)
5. Escalation pipeline:
   - `run_inference_batch.py --source oof` -> `reports/tables/escalation_table_oof.csv` (uses selected ensemble probabilities when available)
   - `step2_feature_builder.py` -> `reports/tables/step2_meta_dataset.csv`
   - `step2_train_meta_model.py` -> CV out-of-sample meta risk scores in `threshold_data.csv`
     and stratified bootstrap evaluation in `step2_meta_bootstrap_metrics.csv`
   - `run_step3_thresholds.py` -> sensitivity-aware threshold optimization outputs:
     `threshold_analysis.csv`, `threshold_analysis_by_human_accuracy.csv`,
     `threshold_sensitivity_summary.csv`, and `artifacts/step3_best_threshold.json`

## Commands
Run from `src/`:

```bash
pip install -r requirements.txt
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

## Important Conventions
- Reported model quality should come from OOF/CV predictions.
- Class mapping remains fixed: `SCD=0`, `MCI=1`, `AD=2`.
- Ensemble = mean of CatBoost + RF + NN probabilities.
- For compatibility, `run_inference_batch.py --source testset` supports the legacy holdout path.
- Tests are standalone `assert` scripts in `tests/`.
