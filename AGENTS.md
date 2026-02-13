# Repository Guidelines

## Project Structure & Module Organization
Primary code is in `src/`:
- Core pipeline: `merge_files.py`, `EDA.py`, `Preprocess.py`, `ModelsFinal.py`, `run_evaluation.py`
- Evaluation framework: `src/eval/`
- HITL escalation: `src/escalation/`
- Shared constants/helpers: `src/utils/`

Data/artifact layout:
- Raw inputs: `Study_files/`
- Processed data: `data/`
- Model artifacts: `artifacts/`
- Results: `Outputs/` and `reports/`
- Tests: `tests/`

## Build, Test, and Development Commands
Run from `src/` unless stated otherwise.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python merge_files.py
python EDA.py
python Preprocess.py
python ModelsFinal.py
python run_evaluation.py
python .\escalation\run_inference_batch.py
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
```

Tests (from repo root):
```powershell
python tests/test_metrics.py
python tests/test_imports.py
```

## Coding Style & Naming Conventions
Use PEP 8, 4-space indentation, `snake_case` for functions/variables, and `UPPER_CASE` for constants. Keep randomness controlled (`numpy`/`tensorflow` seeds) for reproducibility. Use descriptive script names (e.g., `step2_train_meta_model.py`).

## Testing Guidelines
Current tests are standalone assert-based scripts (`tests/test_*.py`), not a full pytest suite. Add deterministic tests with synthetic data and explicit assertions for metric ranges, shapes, and class ordering (`SCD`, `MCI`, `AD`).

## Pipeline Conventions
- Scripts assume working directory is `src/` and use relative paths (`../data`, `../artifacts`, `../reports`).
- `voting_ensemble.pkl` loading may require `escalation/model_stub.py:create_model` bound to `__main__` (see `src/escalation/run_inference_batch.py`).
- Class mapping must remain `SCD=0`, `MCI=1`, `AD=2`.

## Commit & Pull Request Guidelines
Use short imperative commit messages scoped to one logical change. PRs should include changed files, why the change was needed, run commands used for validation, and generated outputs/plots when relevant.

## Documentation Sync Rule
When pipeline behavior changes, update `CLAUDE.md`, `AGENTS.md`, and `README.md` in the same change so contributor and agent guidance stay aligned.
