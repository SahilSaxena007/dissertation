# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions

When starting a new session refer to `MASTERPLAN.md` for context, or ask: `let's continue with the masterplan`.

## Documentation Sync Rule
Whenever pipeline logic, paths, execution commands, or workflow assumptions change, update all three files together in the same change:
- `CLAUDE.md`
- `AGENTS.md`
- `README.md`

## Project Overview

Dissertation project: **Human-in-the-Loop (HITL) framework for Alzheimer's disease classification** using ADNI biomarker data. The system classifies patients into three diagnostic stages (**SCD**, **MCI**, **AD**) using an ensemble of CatBoost, Random Forest, and Neural Network models, then decides whether each prediction should be escalated to a clinician for review.

## Pipeline Architecture

The project follows a sequential multi-step pipeline. All scripts assume `src/` as the working directory and use relative paths (`../artifacts`, `../data`, `../reports`).

### Data Flow

1. **merge_files.py** - Merges raw ADNI CSVs from `Study_files/` into `data/merged_data.csv`
2. **Preprocess.py** - Cleans biomarkers, engineers ratio features, handles outliers (IQR), encodes categoricals, exports `data/preprocessed_data.csv`
3. **ModelsFinal.py** - Trains CatBoost + RF + NN via RandomizedSearchCV, saves all models and train/test splits to `artifacts/`
4. **run_evaluation.py** - Loads models and runs the 11-component evaluation orchestrator on each model
5. **escalation/** pipeline (Steps 2-3 of HITL):
   - `run_inference_batch.py` - Runs escalation engine on test set, produces `escalation_table_testset.csv`
   - `step2_feature_builder.py` - Builds reliability meta-features (7 risk signals) from escalation table
   - `step2_train_meta_model.py` - Trains LogisticRegression meta-model to predict AI error probability
   - `run_step3_thresholds.py` - Optimizes review threshold under a budget constraint
   - `step2_new_patient_meta.py` - Demo: applies full HITL pipeline to a single patient

### Key Modules

- **src/eval/** - 11-component evaluation framework (metrics, visualizations, uncertainty, statistical inference, error taxonomy, explainability/SHAP, bias diagnostics, reporting). Orchestrated by `orchestrator.py:analyze_model_performance()`.
- **src/escalation/** - HITL escalation logic. `escalation_engine.py` computes per-patient signals (uncertainty, model disagreement, missingness, multimodal mismatch). `escalation_rules.py` contains the low-level rule functions. `escalation_config.py` holds tunable thresholds as dataclasses.
- **src/utils/** - Shared constants (`CLASS_NAMES = ["SCD", "MCI", "AD"]`, feature lists, path constants) and data helpers.

## Commands

All commands run from `src/`:

```bash
# Activate venv (Windows)
.venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Full pipeline (sequential)
python merge_files.py
python EDA.py
python Preprocess.py
python ModelsFinal.py
python run_evaluation.py

# HITL escalation pipeline
python .\escalation\run_inference_batch.py
python .\escalation\step2_feature_builder.py
python .\escalation\step2_train_meta_model.py
python .\escalation\run_step3_thresholds.py

# Single patient demo
python .\escalation\run_inference_single.py 42
python .\escalation\step2_new_patient_meta.py 42

# Run tests (from project root)
python tests/test_metrics.py
python tests/test_imports.py
```

## Important Conventions

- **Working directory**: All `src/` scripts use relative paths assuming CWD is `src/`.
- **Model unpickling**: `voting_ensemble.pkl` contains a SciKeras KerasClassifier. Loading it requires `model_stub.py:create_model` patched into `__main__` before `joblib.load()` (see `run_inference_batch.py`).
- **Three diagnostic classes**: Always `['SCD', 'MCI', 'AD']` mapped to `[0, 1, 2]`.
- **12 selected features**: After SelectKBest, model input shape for NN is `(12,)`.
- **Ensemble**: Soft voting = mean of CatBoost + RF + NN probabilities.
- **Escalation levels**: `AI-Autonomous`, `AI-Assisted`, `Clinician-Mandatory`.
- **Artifacts and reports**: Models and arrays in `artifacts/`; CSVs/figures in `Outputs/` and `reports/`.
- **Tests**: Standalone `assert` scripts, not an integrated pytest suite.
