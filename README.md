# dissertation

Human-in-the-loop (HITL) Alzheimer's classification pipeline using ADNI biomarkers. The system predicts `SCD` / `MCI` / `AD` with an ensemble (CatBoost, Random Forest, Neural Network), then applies escalation logic for clinician review.

## Repository Layout
- `src/`: all runnable pipeline scripts
- `Study_files/`: raw ADNI exports
- `data/`: merged and preprocessed CSVs
- `artifacts/`: trained models, splits, scalers, selectors
- `Outputs/`, `reports/`: tables and figures
- `tests/`: standalone validation scripts

## Environment Setup
Run from `src/`.

```powershell
cd src
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## End-to-End Pipeline
Run in this order from `src/`:

```powershell
python merge_files.py
python EDA.py
python Preprocess.py
python KNN_test.py
python selectkbest_test.py
python ModelsFinal.py
python run_evaluation.py
```

### What each step does
1. `merge_files.py`: merge raw files from `../Study_files/` to `../data/merged_data.csv`
2. `EDA.py`: missingness/outlier/correlation analysis and plots
3. `Preprocess.py`: cleaning, encoding, ratio features, outlier capping, `../data/preprocessed_data.csv`
4. `KNN_test.py`: imputation sensitivity analysis
5. `selectkbest_test.py`: feature-count sensitivity analysis
6. `ModelsFinal.py`: train ensemble models and export artifacts
7. `run_evaluation.py`: generate multi-component performance/evaluation reports

## HITL Escalation Pipeline
After model training, run from `src/`:

```powershell
python .\escalation\run_inference_batch.py
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py
```

Optional single-patient demo:

```powershell
python .\escalation\run_inference_single.py 42
python .\escalation\step2_new_patient_meta.py 42
```

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
- Scripts rely on relative paths and expect CWD=`src/`.
- Class mapping is fixed: `SCD=0`, `MCI=1`, `AD=2`.
- Keep documentation synced when pipeline behavior changes: `README.md`, `AGENTS.md`, `CLAUDE.md`.
