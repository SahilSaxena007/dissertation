# dissertation

Human-in-the-loop (HITL) Alzheimer's classification pipeline using ADNI biomarkers. The system predicts `SCD` / `MCI` / `AD` with an ensemble (CatBoost, Random Forest, Neural Network, XGBoost, SVM), then applies escalation logic for clinician review.

## Repository Layout

- `src/`: runnable pipeline scripts
- `src/escalation/`: escalation engine, SHAP utilities, config, rules
- `src/dashboard/`: Streamlit clinician dashboard
- `src/hitl/`: feedback loop, simulated clinician, experiment runner
- `src/evaluation/`: statistical tests, cost analysis, ablation, fairness, uncertainty quantification
- `Study_files/`: raw ADNI exports
- `data/`: merged and preprocessed CSVs
- `artifacts/`: models, preprocessors, OOF/test arrays, SHAP caches, NaN masks
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

## Execution Flow For New Implementers

Use this as the default implementation order if you are running the project for the first time.

### Phase 0: Data + environment readiness

1. Confirm raw ADNI CSVs are present in `../Study_files/`.
2. Create and activate a virtual environment in `src/`.
3. Install dependencies with `pip install -r requirements.txt`.

### Phase 1: Build training dataset

Run from `src/`:

```powershell
python merge_files.py
python EDA.py
python Preprocess.py
```

Checkpoint:
- `../data/merged_data.csv`
- `../data/preprocessed_data.csv`

### Phase 2: Train models and create core artifacts

Run from `src/`:

```powershell
python ModelsFinal.py
python run_evaluation.py
```

What this does:
- Trains 5 base models (CatBoost, RF, NN, XGBoost, SVM) on train split only.
- Produces OOF predictions for leak-free evaluation.
- Builds calibrated ensemble probabilities.
- Saves locked holdout artifacts for unseen demo.

Checkpoint:
- `../artifacts/oof_ens_proba_selected.npy`
- `../Outputs/overall_metrics.csv`

### Phase 3: Build HITL escalation policy (preferred one-command flow)

Run from `src/`:

```powershell
python .\escalation\run_phase2_hitl.py --run-testset-demo
```

What this does internally:
1. Builds OOF escalation table.
2. Builds Step 2 meta-dataset (`ai_error` target + risk features).
3. Trains Step 2 reliability model (`P(AI wrong)`).
4. Optimizes Step 3 threshold under review budget with human-accuracy sensitivity.
5. Optionally evaluates on strict holdout testset demo.

Checkpoint:
- `../reports/tables/escalation_table_oof.csv`
- `../reports/tables/step2_meta_dataset.csv`
- `../artifacts/step2_meta_model.pkl`
- `../artifacts/step3_best_threshold.json`
- `../artifacts/phase2_hitl_summary.json`

### Phase 4: Clinician-facing app

Run from `src/`:

```powershell
streamlit run .\dashboard\app.py
```

Use the app to:
- Review retrospective cases (default source is holdout/testset).
- Inspect explanations (real SHAP when available, surrogate fallback otherwise).
- Submit clinician feedback (stored in `../artifacts/hitl_feedback.db`).
- Run new-patient inference with full pipeline (imputer + selector + scaler + ensemble + meta-risk + threshold).

### Phase 5: Optional post-hoc analyses

Run from `src/`:

```powershell
python .\hitl\experiment_runner.py --accuracies 0.85 0.9 0.95
python evaluation\statistical_tests.py
python evaluation\cost_analysis.py
python evaluation\ablation_study.py
python evaluation\fairness_analysis.py
python evaluation\efficiency_analysis.py
python evaluation\agreement_analysis.py
python evaluation\uncertainty_quantification.py
```

Outputs are written mainly to `../reports/tables/`.

### Minimal validation path (recommended for first run)

Run from repository root:

```powershell
python scripts/run_tests.py
```

If these pass and Phases 1-4 complete, the implementation is operational end-to-end.

### Implementation Validation (single command)

Run from `src/`:

```powershell
python run_validation_pipeline.py
```

This executes key HITL/evaluation scripts and verifies required `reports/tables/*.csv` outputs exist and are non-empty.

### Deterministic Full Rebuild (single command)

Run from `src/`:

```powershell
python run_deterministic_pipeline.py
```

This performs a deterministic end-to-end rebuild (data prep -> training -> HITL Phase 2 -> validation checks).

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
python .\escalation\run_phase2_hitl.py --run-testset-demo
streamlit run .\dashboard\app.py
python .\escalation\run_inference_batch.py --source oof
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
python .\hitl\experiment_runner.py --accuracies 0.85 0.9 0.95
python evaluation\statistical_tests.py
python evaluation\cost_analysis.py
python evaluation\ablation_study.py
python evaluation\fairness_analysis.py
python evaluation\efficiency_analysis.py
python evaluation\agreement_analysis.py
python evaluation\uncertainty_quantification.py
```

### Step outputs

1. `merge_files.py` -> `../data/merged_data.csv`
2. `Preprocess.py` -> `../data/preprocessed_data.csv`
3. `ModelsFinal.py` ->
   - Repeated stratified CV fold metrics: `../Outputs/cv_fold_metrics.csv`
   - CV summary metrics (mean/std): `../Outputs/overall_metrics.csv`
   - OOF probabilities/labels (`../artifacts/oof_*.npy`) for 5 models (CatBoost, RF, NN, XGBoost, SVM)
   - Stacking: `../artifacts/oof_stacked_proba.npy`, `../artifacts/stacking_meta_learner.pkl`, `../artifacts/stacking_comparison.json`
   - Calibration artifacts:
     - `../artifacts/oof_ens_proba_calibrated_sigmoid.npy`
     - `../artifacts/oof_ens_proba_calibrated_isotonic.npy`
     - `../artifacts/oof_ens_proba_selected.npy` (auto-selected from raw/sigmoid/isotonic by OOF CV log-loss, tie-break by ECE)
     - `../artifacts/oof_ens_proba_calibrated.npy` (compatibility alias to selected artifact)
     - `../artifacts/calibration_summary.json`
   - Pre-imputation NaN masks: `../artifacts/nan_mask_oof.npy`, `../artifacts/nan_mask_holdout.npy`
   - `X_oof.npy` computed from global imputer/selector/scaler pipeline (not averaged across folds)
4. `run_evaluation.py` -> uses OOF artifacts by default for leak-free evaluation
5. `run_inference_batch.py --source oof` -> `../reports/tables/escalation_table_oof.csv`
   - Uses selected ensemble probabilities, real SHAP values, pre-imputation NaN masks, populated multimodal feature groups
   - SHAP values cached as `../artifacts/shap_oof_*.npy`, `../artifacts/shap_holdout_*.npy`
6. `step2_feature_builder.py` -> `../reports/tables/step2_meta_dataset.csv`
7. `step2_train_meta_model.py` ->
   - CV out-of-sample risk scores in `../reports/tables/threshold_data.csv`
   - Stratified bootstrap evaluation summary in `../reports/tables/step2_meta_bootstrap_metrics.csv`
8. `run_step3_thresholds.py` ->
   - `../reports/tables/threshold_analysis.csv` (expected policy metrics under human-accuracy sensitivity)
   - `../reports/tables/threshold_analysis_by_human_accuracy.csv` (full per-threshold, per-scenario sweep)
   - `../reports/tables/threshold_sensitivity_summary.csv` (best threshold per human-accuracy scenario)
   - `../artifacts/step3_best_threshold.json` (robust threshold chosen from sensitivity-aware expected accuracy)
9. `run_phase2_hitl.py --run-testset-demo` ->
   - complete OOF-first Phase 2 build and `../artifacts/phase2_hitl_summary.json`
10. `streamlit run .\dashboard\app.py` ->
   - 4 HITL views: queue, detail/review (with SHAP attribution), analytics, feedback history
   - interaction log in `../artifacts/hitl_feedback.db` (SQLite)
   - supports `New Patient Inference` mode (manual or CSV) using trained ensemble + full imputer/selector/scaler pipeline + Step2 meta-risk + Step3 threshold
   - retrospective review defaults to unseen holdout `testset`
11. `python .\hitl\experiment_runner.py --accuracies 0.85 0.9 0.95` ->
   - `../reports/tables/hitl_simulated_clinician_experiment.csv`
   - Feedback loop retrains actual CatBoost+RF+NN ensemble (not surrogate)
12. Evaluation scripts ->
   - `evaluation/statistical_tests.py` -> `../reports/tables/statistical_tests.csv` (McNemar, bootstrap AUC-difference test, bootstrap CIs)
   - `evaluation/cost_analysis.py` -> `../reports/tables/cost_analysis.csv` (clinical cost-weighted accuracy)
   - `evaluation/ablation_study.py` -> `../reports/tables/ablation_study.csv` (meta-model feature ablation)
   - `evaluation/fairness_analysis.py` -> `../reports/tables/fairness_analysis.csv`, `../reports/tables/fairness_summary.csv` (per-group + disparity summary metrics)
   - `evaluation/efficiency_analysis.py` -> `../reports/tables/efficiency_analysis.csv`, `../reports/tables/efficiency_summary.csv` (decision-time and interaction metrics from HITL logs)
   - `evaluation/agreement_analysis.py` -> `../reports/tables/agreement_analysis.csv`, `../reports/tables/agreement_summary.csv` (AI-human agreement/disagreement by class/escalation reason)
   - `evaluation/uncertainty_quantification.py` -> `../reports/tables/conformal_prediction.csv`, `../reports/tables/uncertainty_decomposition.csv`

## Optional Legacy Mode

`python .\escalation\run_inference_batch.py --source testset` runs holdout testset escalation with calibration applied.

## Tests

Run from repository root:

```powershell
python scripts/run_tests.py
```

This runs the full test suite with UTF-8 output and automatically prefers `src/.venv/Scripts/python.exe` when available.

## Notes

- Scripts use relative paths and expect CWD=`src/`.
- Class mapping is fixed: `SCD=0`, `MCI=1`, `AD=2`.
- Ensemble uses 5 models (CatBoost, RF, NN, XGBoost, SVM). Stacking meta-learner used when it outperforms simple average.
- Phase 2 training should use OOF artifacts; testset remains demo/validation.
- Keep docs synced after pipeline changes: `README.md`, `AGENTS.md`, `CLAUDE.md`.
