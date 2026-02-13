# Repository Guidelines

## Project Structure & Module Organization
Primary code is in `src/`: core pipeline scripts (`merge_files.py`, `EDA.py`, `Preprocess.py`, `ModelsFinal.py`, `run_evaluation.py`), evaluation modules in `src/evaluation/`, HITL escalation logic in `src/escalation/`, dashboard in `src/dashboard/`, and feedback loop in `src/hitl/`.

Data and outputs:
- `Study_files/` raw CSVs
- `data/` merged/preprocessed data
- `artifacts/` models and OOF/test artifacts
- `Outputs/` and `reports/` figures/tables
- `tests/` standalone test scripts

## Build, Test, and Development Commands
Run from `src/` unless noted.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python merge_files.py
python EDA.py
python Preprocess.py
python ModelsFinal.py
python run_evaluation.py
python .\escalation\run_phase2_hitl.py --run-testset-demo
streamlit run .\dashboard\app.py
python .\hitl\experiment_runner.py --accuracies 0.85 0.9 0.95
python .\escalation\run_inference_batch.py --source oof
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
python evaluation\statistical_tests.py
python evaluation\cost_analysis.py
python evaluation\ablation_study.py
python evaluation\fairness_analysis.py
python evaluation\uncertainty_quantification.py
```

Tests (repo root):
```powershell
python tests/test_imports.py
python tests/test_metrics.py
```

## Coding Style & Naming Conventions
Use PEP 8, 4-space indentation, `snake_case` names, and `UPPER_CASE` constants. Keep training deterministic with explicit seeds. Prefer descriptive filenames (for example, `step2_train_meta_model.py`).

## Testing Guidelines
Tests are assert-based scripts (`tests/test_*.py`) rather than a unified pytest suite. Add deterministic tests with synthetic data and strict assertions for metric ranges, output shapes, and class order (`SCD`, `MCI`, `AD`).

## Pipeline Conventions
- Scripts assume CWD=`src/` and use relative paths (`../data`, `../artifacts`, `../reports`).
- Reported performance must use repeated stratified OOF/CV outputs and include fold-distributed summary (mean/std), not same-split train/eval.
- `ModelsFinal.py` must create a strict locked holdout split (`X_test.npy`, `y_test.npy`) and train fitting must remain train-split only.
- `ModelsFinal.py` trains 5 models: CatBoost, RandomForest, Neural Network, XGBoost, SVM. Ensemble = mean of all 5 model probabilities. Stacking meta-learner (LogisticRegression on OOF probs) used when it outperforms simple average.
- `ModelsFinal.py` computes `X_oof.npy` from the global imputer/selector/scaler pipeline (not averaged across folds).
- `ModelsFinal.py` saves pre-imputation NaN masks (`nan_mask_oof.npy`, `nan_mask_holdout.npy`) projected through feature selector.
- `run_evaluation.py` should consume OOF artifacts when available.
- `run_phase2_hitl.py` is the preferred one-command Phase 2 runner and should remain OOF-first.
- Escalation pipeline uses real SHAP values (TreeExplainer for CatBoost, RF, XGBoost), pre-imputation NaN masks for missingness, and populated multimodal feature groups via `build_escalation_config()`.
- `run_inference_batch.py --source testset` applies calibration to testset ensemble probs.
- `dashboard/app.py` is the clinician-facing Streamlit interface; loads imputer + selector for new-patient full pipeline inference; uses real SHAP when available, falls back to surrogate; indexes raw_features using train/holdout indices.
- `hitl/feedback_loop.py` retrains actual CatBoost+RF+NN ensemble (not LogisticRegression surrogate); supports `fast=True` for dashboard interactive use.
- `ModelsFinal.py` writes OOF ensemble calibration artifacts (`oof_ens_proba_calibrated_*.npy`, `oof_ens_proba_selected.npy`, `oof_ens_proba_calibrated.npy`, `calibration_summary.json`) and downstream OOF escalation should prefer selected ensemble probabilities.
- Step 3 thresholding should use human-accuracy sensitivity analysis (default range `0.85-0.95`) rather than a single fixed clinician-accuracy value.
- Step 2 meta-model evaluation should include stratified bootstrap uncertainty estimates (`step2_meta_bootstrap_metrics.csv`).
- Evaluation scripts in `src/evaluation/` provide statistical tests, cost analysis, ablation study, fairness analysis, and conformal prediction/uncertainty decomposition.
- Keep class mapping fixed: `SCD=0`, `MCI=1`, `AD=2`.

## Commit & Pull Request Guidelines
Use short imperative commit messages scoped to one logical change. PRs should include changed files, reason for change, validation commands run, and produced outputs/plots when relevant.

## Documentation Sync Rule
When pipeline behavior, artifact names, or run commands change, update `CLAUDE.md`, `AGENTS.md`, and `README.md` in the same change.
