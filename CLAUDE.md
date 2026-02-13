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
   - OOF artifacts for leak-free evaluation on train split: `oof_cat_proba.npy`, `oof_rf_proba.npy`, `oof_nn_proba.npy`, `oof_xgb_proba.npy`, `oof_svm_proba.npy`, `oof_ens_proba.npy`, `X_oof.npy`, `y_oof.npy`
   - stacking artifacts: `oof_stacked_proba.npy`, `stacking_meta_learner.pkl`, `stacking_comparison.json`
   - calibration artifacts: `oof_ens_proba_calibrated_sigmoid.npy`, `oof_ens_proba_calibrated_isotonic.npy`, `oof_ens_proba_selected.npy`, `oof_ens_proba_calibrated.npy` (compat alias), `calibration_summary.json`, calibrator pickles
   - pre-imputation NaN masks: `nan_mask_oof.npy`, `nan_mask_holdout.npy`
   - deployment artifacts: models (CatBoost, RF, NN, XGBoost, SVM), imputer, selector, scaler
   - strict holdout split arrays: `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `train_indices.npy`, `holdout_indices.npy`
   - `X_oof.npy` is computed from the global imputer/selector/scaler pipeline (not averaged across folds)
4. `run_evaluation.py` -> evaluates with OOF artifacts by default (falls back to strict holdout split)
5. Escalation pipeline:
   - `run_phase2_hitl.py` (one-command full Phase 2 runner, OOF-first)
   - `run_inference_batch.py --source oof` -> `reports/tables/escalation_table_oof.csv` (uses selected ensemble probabilities, real SHAP, pre-imputation NaN mask, populated multimodal groups)
   - `escalation/shap_utils.py` -> computes and caches SHAP values (`shap_oof_*.npy`, `shap_holdout_*.npy`)
   - `escalation/escalation_config.py` -> `build_escalation_config()` populates multimodal feature groups from `selected_features.json`
   - `step2_feature_builder.py` -> `reports/tables/step2_meta_dataset.csv`
   - `step2_train_meta_model.py` -> CV out-of-sample meta risk scores in `threshold_data.csv`
     and stratified bootstrap evaluation in `step2_meta_bootstrap_metrics.csv`
   - `run_step3_thresholds.py` -> sensitivity-aware threshold optimization outputs:
     `threshold_analysis.csv`, `threshold_analysis_by_human_accuracy.csv`,
     `threshold_sensitivity_summary.csv`, and `artifacts/step3_best_threshold.json`
   - `run_phase2_hitl.py` also writes `artifacts/phase2_hitl_summary.json`
6. HITL app + feedback loop:
   - `dashboard/app.py` (Streamlit clinician dashboard with queue, detail/review, analytics, feedback history, and new-patient inference)
   - `artifacts/hitl_feedback.db` (SQLite interaction log)
   - `hitl/feedback_loop.py` (active-learning retraining with actual CatBoost+RF+NN ensemble, `--fast` mode for dashboard)
   - `hitl/experiment_runner.py` (simulated clinician experiment sweep)
7. Evaluation scripts (from `src/`):
   - `evaluation/statistical_tests.py` -> `reports/tables/statistical_tests.csv` (McNemar, DeLong, bootstrap CIs)
   - `evaluation/cost_analysis.py` -> `reports/tables/cost_analysis.csv` (clinical cost-weighted accuracy)
   - `evaluation/ablation_study.py` -> `reports/tables/ablation_study.csv` (meta-model feature ablation)
   - `evaluation/fairness_analysis.py` -> `reports/tables/fairness_analysis.csv` (per-group metrics, demographic parity)
   - `evaluation/uncertainty_quantification.py` -> `reports/tables/conformal_prediction.csv`, `reports/tables/uncertainty_decomposition.csv`

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

## Important Conventions
- Reported model quality should come from OOF/CV predictions.
- Class mapping remains fixed: `SCD=0`, `MCI=1`, `AD=2`.
- Ensemble = mean of CatBoost + RF + NN + XGBoost + SVM probabilities (5 models). Stacking meta-learner used when it outperforms simple average.
- Selected calibration artifact used for escalation uncertainty signals.
- Phase 2 should train on OOF artifacts to avoid leakage; `--source testset` is demonstration only.
- `run_inference_batch.py --source testset` is the unseen holdout demo path (untouched 20% split); calibration is applied to testset ensemble probs.
- Escalation uses real SHAP values (TreeExplainer), pre-imputation NaN masks, and populated multimodal feature groups.
- Dashboard uses real SHAP feature attribution when available, falling back to surrogate.
- Feedback loop retrains actual CatBoost+RF+NN ensemble (not LogisticRegression surrogate).
- Tests are standalone `assert` scripts in `tests/`.
