# Repository Guidelines

## Project Structure & Module Organization
Primary code is in `src/`: core pipeline scripts (`merge_files.py`, `EDA.py`, `Preprocess.py`, `ModelsFinal.py`, `run_evaluation.py`), evaluation modules in `src/eval/`, and HITL escalation logic in `src/escalation/`. Shared constants/helpers are in `src/utils/`.

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
python .\escalation\run_inference_batch.py --source oof
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
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
- `run_evaluation.py` should consume OOF artifacts when available.
- `ModelsFinal.py` writes OOF ensemble calibration artifacts (`oof_ens_proba_calibrated_*.npy`, `oof_ens_proba_selected.npy`, `oof_ens_proba_calibrated.npy`, `calibration_summary.json`) and downstream OOF escalation should prefer selected ensemble probabilities.
- `run_inference_batch.py --source testset` is the unseen holdout demo path.
- Step 3 thresholding should use human-accuracy sensitivity analysis (default range `0.85-0.95`) rather than a single fixed clinician-accuracy value.
- Step 2 meta-model evaluation should include stratified bootstrap uncertainty estimates (`step2_meta_bootstrap_metrics.csv`).
- Keep class mapping fixed: `SCD=0`, `MCI=1`, `AD=2`.

## Commit & Pull Request Guidelines
Use short imperative commit messages scoped to one logical change. PRs should include changed files, reason for change, validation commands run, and produced outputs/plots when relevant.

## Documentation Sync Rule
When pipeline behavior, artifact names, or run commands change, update `CLAUDE.md`, `AGENTS.md`, and `README.md` in the same change.
