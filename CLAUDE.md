# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
# Activate venv
src/.venv/Scripts/activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Full pipeline (sequential)
python merge_files.py
python Preprocess.py
python ModelsFinal.py
python run_evaluation.py

# HITL escalation pipeline (from src/)
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

- **Working directory**: All `src/` scripts use relative paths assuming CWD is `src/`. The escalation scripts additionally assume CWD is `src/escalation/` for their internal imports (they use bare imports like `from escalation_config import ...`).
- **Model unpickling**: `voting_ensemble.pkl` contains a SciKeras KerasClassifier. Loading it requires `model_stub.py:create_model` to be patched into `__main__` before `joblib.load()`. See `run_inference_batch.py` for the pattern.
- **Three diagnostic classes**: Always `["SCD", "MCI", "AD"]` mapped to `[0, 1, 2]`. Defined in both `utils/constants.py` and `escalation/escalation_config.py`.
- **12 selected features**: After SelectKBest, the model uses 12 features (see `artifacts/selected_features.json`). Input shape for the NN stub is `(12,)`.
- **Ensemble**: Soft voting = simple mean of CatBoost + RF + NN predicted probabilities.
- **Escalation levels**: Three tiers - "AI-Autonomous" (no flags), "AI-Assisted" (soft flags like disagreement/low confidence), "Clinician-Mandatory" (hard flags like critical missing data or multimodal mismatch).
- **Artifacts**: Serialized models (`.pkl`, `.h5`), numpy arrays (`.npy`), and threshold config (`.json`) live in `artifacts/`. Pipeline outputs (CSVs, figures) go to `reports/` and `Outputs/`.
- **No test framework**: Tests are standalone Python scripts using `assert` statements, not pytest.
