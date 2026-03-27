# Alzheimer's HITL Classification Pipeline

Human-in-the-loop (HITL) framework for Alzheimer's disease staging using ADNI biomarkers.
The system classifies patients as **SCD / MCI / AD** using an ensemble model, then decides
whether each case should be handled autonomously by the AI or escalated to a clinician for review.

---

## Project Structure

```
.
├── README.md
├── data/                        # Intermediate and processed datasets
│   ├── merged_data.csv
│   └── preprocessed_data.csv
├── Study_files/                 # Raw ADNI exports (not tracked in git)
├── artifacts/                   # Trained models, OOF arrays, thresholds (not tracked in git)
├── reports/                     # Generated figures and tables (not tracked in git)
├── Outputs/                     # Plots from base pipeline scripts (not tracked in git)
└── src/
    ├── merge_files.py           # Step 1: merge raw ADNI CSVs
    ├── EDA.py                   # Step 2: exploratory data analysis
    ├── Preprocess.py            # Step 3: cleaning and feature engineering
    ├── KNN_test.py              # Step 4: KNN imputation sensitivity analysis
    ├── selectkbest_test.py      # Step 5: feature count sensitivity analysis
    ├── ModelsFinal.py           # Step 6: train ensemble + save artifacts
    ├── run_pipeline.py          # Convenience: runs steps 1-6 in sequence
    ├── run_evaluation.py        # Runs model_eval on OOF artifacts
    ├── run_analysis_suite.py    # Runs the full post-hoc analysis suite
    ├── run_all.py               # End-to-end rebuild (data -> training -> HITL -> analysis)
    ├── requirements.txt
    ├── escalation/              # HITL escalation engine (Steps 1-3)
    │   ├── run_escalation.py    # Main entry point for Phase 2 HITL build
    │   ├── inference_batch.py   # OOF and testset escalation tables
    │   ├── inference_single.py  # Single-patient inference demo
    │   ├── engine.py            # Rule-based escalation logic
    │   ├── meta_feature_builder.py  # Builds risk features for meta-model
    │   ├── meta_model_trainer.py    # Trains Step 2 reliability model
    │   ├── meta_model_inference.py  # Applies meta-model to new patients
    │   ├── threshold_optimizer.py   # Step 3 threshold optimisation
    │   ├── threshold_runner.py      # Standalone Step 3 runner
    │   ├── config.py            # Escalation thresholds and class names
    │   └── rules.py             # Escalation rule definitions
    ├── dashboard/
    │   └── app.py               # Streamlit clinician dashboard
    ├── hitl/
    │   └── run_experiment.py    # Simulated clinician experiment
    ├── evaluation/
    │   ├── run_analysis.py      # Runs all 7 evaluation scripts in sequence
    │   ├── statistical_tests.py
    │   ├── cost_analysis.py
    │   ├── ablation_study.py
    │   ├── fairness_analysis.py
    │   ├── uncertainty_quantification.py
    │   ├── efficiency_analysis.py
    │   └── agreement_analysis.py
    ├── model_eval/              # 11-component model evaluation framework
    └── utils/                   # Shared utilities and project config
```

---

## Setup

All scripts assume CWD is `src/`. Run all commands from there.

```bash
cd src
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## Data

- **`../Study_files/`** — Raw ADNI CSVs (e.g. `ADNIMERGE_14Oct2024.csv`). Must be present before running `merge_files.py`.
- **`../data/`** — Intermediate datasets produced by the pipeline.
- **`../artifacts/`** — Trained models, OOF arrays, thresholds. Produced during training; required by all downstream steps.
- **`../reports/tables/`** and **`../reports/figures/`** — All evaluation outputs.
- **`../Outputs/`** — Figures and metrics from the base model scripts.

---

## Pipeline

### Stage 1 — Data preparation (Benjamin's work)

Run from `src/`:

```bash
python merge_files.py
python EDA.py
python Preprocess.py
```

| Script           | Input                  | Output                                                     |
| ---------------- | ---------------------- | ---------------------------------------------------------- |
| `merge_files.py` | `../Study_files/*.csv` | `../data/merged_data.csv`                                  |
| `EDA.py`         | `merged_data.csv`      | Figures in `../Outputs/`                                   |
| `Preprocess.py`  | `merged_data.csv`      | `../data/preprocessed_data.csv`, boxplots in `../Outputs/` |

Optionally run sensitivity analyses before training:

```bash
python KNN_test.py          # effect of KNN k on downstream performance
python selectkbest_test.py  # effect of feature count on downstream performance
```

---

### Stage 2 — Model training (Benjamin's work)

```bash
python ModelsFinal.py
```

Trains CatBoost, Random Forest, Neural Network, XGBoost, and SVM on the training split.
Produces out-of-fold (OOF) predictions for all models, a calibrated voting ensemble,
and a locked holdout test set. Key artifacts saved to `../artifacts/`:

- `voting_ensemble.pkl` — trained ensemble
- `oof_ens_proba_selected.npy` — calibrated OOF ensemble probabilities
- `X_oof.npy`, `X_test.npy`, `y_oof.npy`, `y_test.npy` — arrays for downstream steps
- `nan_mask_oof.npy`, `nan_mask_holdout.npy` — pre-imputation missingness masks
- `calibration_summary.json` — which calibration method was selected and why

---

### Stage 3 — Model evaluation

```bash
python run_evaluation.py
```

Runs the 11-component evaluation framework (`model_eval/`) on the OOF artifacts.
Outputs per-model metrics, confusion matrices, ROC curves, calibration curves,
bootstrap CIs, SHAP feature importance, and error taxonomy to `../reports/`.

---

### Stage 4 — HITL escalation policy

```bash
python escalation/run_escalation.py --run-testset-demo
```

Builds the full Phase 2 HITL pipeline on OOF data (leak-free), then optionally
evaluates the resulting policy on the held-out test set.

Internally this runs four steps:

1. **Step 1** — Build OOF escalation table (rule-based engine assigns an escalation level to each sample based on confidence, disagreement, and missing data signals).
2. **Step 2** — Build meta-dataset and train a reliability model that predicts `P(AI wrong)` per sample.
3. **Step 3** — Optimise the escalation threshold `τ*` under a review-budget constraint with human-accuracy sensitivity analysis.
4. **Testset demo** _(optional, `--run-testset-demo`)_ — Evaluate the learned policy on the strict holdout.

Key outputs:

| File                                                  | Description                                |
| ----------------------------------------------------- | ------------------------------------------ |
| `../reports/tables/escalation_table_oof.csv`          | Per-sample escalation decisions (OOF)      |
| `../reports/tables/step2_meta_dataset.csv`            | Risk feature dataset for meta-model        |
| `../artifacts/step2_meta_model.pkl`                   | Trained reliability model                  |
| `../artifacts/step3_best_threshold.json`              | Optimal threshold `τ*` and policy metrics  |
| `../artifacts/phase2_hitl_summary.json`               | Full Phase 2 summary                       |
| `../reports/tables/threshold_analysis.csv`            | Policy metrics across all thresholds       |
| `../reports/tables/threshold_sensitivity_summary.csv` | Best threshold per human-accuracy scenario |

To run Step 3 threshold optimisation in isolation:

```bash
python escalation/threshold_runner.py
```

---

### Stage 5 — Clinician dashboard

```bash
streamlit run dashboard/app.py
```

Opens a browser-based dashboard with four views:

- **Review queue** — cases flagged for clinician review, sorted by risk score.
- **Case detail** — prediction breakdown, SHAP feature attributions, escalation reasons.
- **Analytics** — review rate, accuracy gain, uncertainty distributions.
- **Feedback history** — log of all submitted clinician decisions.

Feedback is stored in `../artifacts/hitl_feedback.db` (SQLite).
The dashboard also supports **New Patient Inference**: enter feature values manually
or upload a CSV and the full pipeline (imputer → selector → scaler → ensemble → meta-risk → threshold) runs end-to-end.

---

### Stage 6 — Post-hoc analysis

```bash
python run_analysis_suite.py
```

Runs the simulated clinician experiment and all evaluation scripts in sequence,
then verifies that all required output CSVs exist and are non-empty.

To run individual analyses:

```bash
python hitl/run_experiment.py --accuracies 0.85 0.9 0.95
python evaluation/statistical_tests.py
python evaluation/cost_analysis.py
python evaluation/ablation_study.py
python evaluation/fairness_analysis.py
python evaluation/uncertainty_quantification.py
python evaluation/efficiency_analysis.py
python evaluation/agreement_analysis.py
```

| Script                          | Output                                                      |
| ------------------------------- | ----------------------------------------------------------- |
| `run_experiment.py`             | `hitl_simulated_clinician_experiment.csv`                   |
| `statistical_tests.py`          | `statistical_tests.csv`                                     |
| `cost_analysis.py`              | `cost_analysis.csv`                                         |
| `ablation_study.py`             | `ablation_study.csv`                                        |
| `fairness_analysis.py`          | `fairness_analysis.csv`                                     |
| `uncertainty_quantification.py` | `conformal_prediction.csv`, `uncertainty_decomposition.csv` |
| `efficiency_analysis.py`        | `efficiency_analysis.csv`                                   |
| `agreement_analysis.py`         | `agreement_analysis.csv`                                    |

All CSVs are written to `../reports/tables/`.

---

## Convenience Scripts

| Script                  | What it does                                               |
| ----------------------- | ---------------------------------------------------------- |
| `run_pipeline.py`       | Runs Stages 1–2 (data prep + training) in one command      |
| `run_analysis_suite.py` | Runs Stage 6 and validates all outputs                     |
| `run_all.py`            | Full end-to-end rebuild: data → training → HITL → analysis |

---

## Notes

- All scripts expect CWD to be `src/`. Relative paths (`../artifacts`, `../data`, etc.) are resolved from there.
- Class mapping is fixed: `SCD = 0`, `MCI = 1`, `AD = 2`.
- The OOF artifacts are used for all HITL training. The held-out test set is only used for the optional demo evaluation.
- Random seeds are fixed throughout for reproducibility.
