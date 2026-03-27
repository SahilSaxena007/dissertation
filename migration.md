# Migration Plan — Project Restructure

## Goals
- Professional file names throughout
- **One runner script per module folder** (the key structural change)
- Rename `eval/` to `model_eval/` so the two evaluation folders are unambiguous
- Zero logic changes — outputs must be byte-for-byte identical
- All imports updated atomically with each file move

---

## Hard Constraints
- Benjamin's files are **completely untouched**: `merge_files.py`, `EDA.py`, `Preprocess.py`, `KNN_test.py`, `selectkbest_test.py`, `ModelsFinal.py`, `README_Benjamin.md`
- `data/`, `Outputs/`, `Study_files/` folder structure stays identical
- CWD assumption stays the same: everything still runs from `src/`

---

## Files Being Deleted
| File | Reason |
|---|---|
| `src/escalation/temp.py` | Self-describes as "Legacy debugging scratch file. Deprecated." |

---

## Why `eval/` → `model_eval/`

From `Sahil.md`, Sahil's work has two distinct evaluation stages:
- **"Analysis of the provided AI code"** (Benjamin's models — CatBoost, RF, NN performance, calibration, SHAP) → this is what `eval/` does. Rename to **`model_eval/`**.
- **"Design experiments comparing AI-only vs. HITL performance"** (statistical tests, cost analysis, ablation, fairness, agreement) → this is what `evaluation/` does. Stays as **`evaluation/`**.

The two folders are now unambiguous: `model_eval/` = evaluate the ML models; `evaluation/` = evaluate the HITL system.

---

## Combining Opportunities (Save Time)

| What | Current situation | Proposal |
|---|---|---|
| 8 evaluation scripts | Each run individually in `run_validation_pipeline.py` | Combined into **one new `evaluation/run_analysis.py`** |
| `run_step3_thresholds.py` | Run standalone in `run_validation_pipeline.py` AND called internally by `run_phase2_hitl.py` | **Redundant as standalone** — escalation runner already covers it; keep file for ad-hoc use but remove from pipeline meta-runner |
| Two meta-runners | `run_deterministic_pipeline.py` (everything) and `run_validation_pipeline.py` (analysis only) | Rename to `run_all.py` and `run_analysis_suite.py`; simplify `run_analysis_suite.py` to call the new module runners |

---

## Complete Runner Map (After Migration)

| Runner | Location | What it runs | Replaces |
|---|---|---|---|
| `run_pipeline.py` | `src/` | Benjamin's 6 files (merge → EDA → Preprocess → KNN_test → selectkbest_test → ModelsFinal) | Was `run_deterministic_pipeline.py` (subset) |
| `run_evaluation.py` | `src/` | model_eval orchestrator (base model analysis) | Already exists — stays, just import updated |
| `escalation/run_escalation.py` | `src/escalation/` | Full HITL pipeline: inference → meta-model → threshold optimisation | Was `run_phase2_hitl.py` |
| `hitl/run_experiment.py` | `src/hitl/` | Simulated clinician experiment (AI-only vs HITL comparison) | Was `experiment_runner.py` |
| `evaluation/run_analysis.py` | `src/evaluation/` | All 8 analysis scripts in sequence | **NEW** — was 8 separate subprocess calls in `run_validation_pipeline.py` |
| `dashboard/app.py` | `src/dashboard/` | Streamlit clinician dashboard | Unchanged |

### Meta-runners (convenience, call the above)
| Runner | Location | What it runs |
|---|---|---|
| `run_all.py` | `src/` | Calls ALL module runners end-to-end (full reproducible pipeline) |
| `run_analysis_suite.py` | `src/` | Calls hitl + evaluation runners only (skip retraining) |

---

## 1. `eval/` → `model_eval/` (Rename Folder Only)

All files inside move with **no content changes** except for the one import in `run_evaluation.py`.

### Files (all keep their existing names inside the new folder)

| Old path | New path |
|---|---|
| `eval/__init__.py` | `model_eval/__init__.py` |
| `eval/orchestrator.py` | `model_eval/orchestrator.py` |
| `eval/metrics.py` | `model_eval/metrics.py` |
| `eval/visualizations.py` | `model_eval/visualizations.py` |
| `eval/uncertainty.py` | `model_eval/uncertainty.py` |
| `eval/statistical_inference.py` | `model_eval/statistical_inference.py` |
| `eval/explainability.py` | `model_eval/explainability.py` |
| `eval/bias_diagnostics.py` | `model_eval/bias_diagnostics.py` |
| `eval/error_taxonomy.py` | `model_eval/error_taxonomy.py` |
| `eval/reporting.py` | `model_eval/reporting.py` |
| `eval/performance_analysis1.py` | `model_eval/performance_analysis.py` ← remove the `1` suffix |

### Import updates for this rename

| File | Old import | New import |
|---|---|---|
| `src/run_evaluation.py` | `from eval.orchestrator import analyze_model_performance` | `from model_eval.orchestrator import analyze_model_performance` |
| All `model_eval/*.py` internal imports | `from . import metrics` etc. | ✅ Unchanged — relative imports work as-is |

---

## 2. `evaluation/` — Add Runner (No Renames)

All existing files stay exactly as they are. One new file is added.

### New file: `evaluation/run_analysis.py`

Thin runner that calls all 8 scripts in order (same order as `run_validation_pipeline.py` currently does via subprocess):

```
statistical_tests.py
cost_analysis.py
ablation_study.py
fairness_analysis.py
uncertainty_quantification.py
oof_vs_holdout_analysis.py
agreement_analysis.py
efficiency_analysis.py
```

This is a **new file** with no logic — it just calls `subprocess.run` for each script, same pattern as the existing pipeline runners.

---

## 3. `escalation/` — File Renames

### Rename map

| Old name | New name | Reason |
|---|---|---|
| `escalation_config.py` | `config.py` | Redundant prefix inside `escalation/` folder |
| `escalation_engine.py` | `engine.py` | Redundant prefix |
| `escalation_rules.py` | `rules.py` | Redundant prefix |
| `step2_feature_builder.py` | `meta_feature_builder.py` | Removes implementation step number; describes purpose |
| `step2_train_meta_model.py` | `meta_model_trainer.py` | Same reason |
| `step2_new_patient_meta.py` | `meta_model_inference.py` | Same reason |
| `run_inference_batch.py` | `inference_batch.py` | Drop `run_` prefix — it's a module inside a folder, not a top-level runner |
| `run_inference_single.py` | `inference_single.py` | Same reason |
| `run_phase2_hitl.py` | `run_escalation.py` | This IS the folder's runner — clearer name |
| `run_step3_thresholds.py` | `threshold_runner.py` | Still runnable standalone; removes step-number naming |
| `test_step2_escalation.py` | `test_escalation.py` | Cleaner test name |
| `shap_utils.py` | ✅ stays | Already clear |
| `threshold_optimizer.py` | ✅ stays | Already clear |
| `model_stub.py` | ✅ stays | Already clear |

### Import updates — escalation uses bare imports (CWD = `src/escalation/`)

| File | Old bare import | New bare import |
|---|---|---|
| `engine.py` | `from escalation_config import ...` | `from config import ...` |
| `engine.py` | `from escalation_rules import ...` | `from rules import ...` |
| `rules.py` | `from escalation_config import EscalationConfig` | `from config import EscalationConfig` |
| `inference_batch.py` | `from escalation_engine import ...` | `from engine import ...` |
| `inference_batch.py` | `from escalation_config import ...` | `from config import ...` |
| `inference_single.py` | `from escalation_engine import ...` | `from engine import ...` |
| `inference_single.py` | `from escalation_config import ...` | `from config import ...` |
| `run_escalation.py` | `from run_inference_batch import ...` | `from inference_batch import ...` |
| `run_escalation.py` | `from run_step3_thresholds import build_human_accuracy_grid` | `from threshold_runner import build_human_accuracy_grid` |
| `run_escalation.py` | `from step2_feature_builder import ...` | `from meta_feature_builder import ...` |
| `run_escalation.py` | `from step2_train_meta_model import ...` | `from meta_model_trainer import ...` |
| `threshold_runner.py` | `from threshold_optimizer import ...` | ✅ Unchanged |
| `meta_model_inference.py` | `from escalation_engine import ...` | `from engine import ...` |
| `meta_model_inference.py` | `from escalation_config import ...` | `from config import ...` |
| `test_escalation.py` | `from escalation_engine import ...` | `from engine import ...` |
| `test_escalation.py` | `from escalation_config import ...` | `from config import ...` |

### Package-level imports from outside escalation/

| File | Old import | New import |
|---|---|---|
| `dashboard/app.py` | `from escalation.escalation_config import ...` | `from escalation.config import ...` |
| `dashboard/app.py` | `from escalation.escalation_engine import ...` | `from escalation.engine import ...` |
| `dashboard/app.py` | `from escalation.model_stub import ...` | ✅ Unchanged |

---

## 4. `hitl/` — Rename One File

| Old name | New name | Reason |
|---|---|---|
| `experiment_runner.py` | `run_experiment.py` | Consistent: `run_` prefix marks it as the folder's runner |
| `feedback_loop.py` | ✅ stays | |
| `simulated_clinician.py` | ✅ stays | |
| `interaction_logger.py` | ✅ stays | |

### Import updates

| File | Old reference | New reference |
|---|---|---|
| `run_experiment.py` internal | `from .feedback_loop import ...` | ✅ Unchanged — relative import |
| `run_analysis_suite.py` (meta-runner) | `hitl/experiment_runner.py` subprocess | `hitl/run_experiment.py` |

---

## 5. Top-Level Runners

| Old name | New name | Notes |
|---|---|---|
| `run_deterministic_pipeline.py` | `run_all.py` | Calls all module runners end-to-end; update subprocess paths inside |
| `run_validation_pipeline.py` | `run_analysis_suite.py` | Simplified: calls `hitl/run_experiment.py` + `evaluation/run_analysis.py`; remove redundant `escalation/run_step3_thresholds.py` call (already inside escalation runner) |
| `run_evaluation.py` | ✅ stays at `src/` level | Only import line changes (`eval` → `model_eval`) |

### Subprocess path updates inside `run_all.py`

| Old call | New call |
|---|---|
| `run_evaluation.py` | `run_evaluation.py` ✅ stays |
| `escalation/run_phase2_hitl.py` | `escalation/run_escalation.py` |
| `run_validation_pipeline.py` | `run_analysis_suite.py` |

### Subprocess path updates inside `run_analysis_suite.py`

| Old call | New call |
|---|---|
| `run_evaluation.py` (--full only) | `run_evaluation.py` ✅ stays |
| `escalation/run_phase2_hitl.py` (--full only) | `escalation/run_escalation.py` |
| `hitl/experiment_runner.py` | `hitl/run_experiment.py` |
| `escalation/run_step3_thresholds.py` | **REMOVED** — already runs inside `escalation/run_escalation.py` |
| `evaluation/statistical_tests.py` ... (8 individual calls) | `evaluation/run_analysis.py` (1 call) |

---

## 6. `dashboard/` and `utils/` — No Changes

| Folder | Decision |
|---|---|
| `dashboard/app.py` | Only escalation import lines updated (see section 3) |
| `utils/` (all files) | ✅ Completely unchanged |

---

## Final Structure After Migration

```
src/
├── merge_files.py                    [UNTOUCHED — Benjamin]
├── EDA.py                            [UNTOUCHED — Benjamin]
├── Preprocess.py                     [UNTOUCHED — Benjamin]
├── KNN_test.py                       [UNTOUCHED — Benjamin]
├── selectkbest_test.py               [UNTOUCHED — Benjamin]
├── ModelsFinal.py                    [UNTOUCHED — Benjamin]
├── README_Benjamin.md                [UNTOUCHED]
├── requirements.txt                  [UNTOUCHED]
│
├── run_pipeline.py                   ← was run_deterministic_pipeline.py (runs Benjamin's 6 files only)
├── run_evaluation.py                 [stays — import updated: eval→model_eval]
├── run_all.py                        ← was run_deterministic_pipeline.py (full end-to-end)
├── run_analysis_suite.py             ← was run_validation_pipeline.py (simplified)
│
├── model_eval/                       ← was eval/ (renamed folder)
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── metrics.py
│   ├── visualizations.py
│   ├── uncertainty.py
│   ├── statistical_inference.py
│   ├── explainability.py
│   ├── bias_diagnostics.py
│   ├── error_taxonomy.py
│   ├── reporting.py
│   └── performance_analysis.py      ← was performance_analysis1.py
│
├── evaluation/                       [folder unchanged — new runner added]
│   ├── run_analysis.py               ← NEW runner
│   ├── statistical_tests.py
│   ├── cost_analysis.py
│   ├── ablation_study.py
│   ├── fairness_analysis.py
│   ├── uncertainty_quantification.py
│   ├── oof_vs_holdout_analysis.py
│   ├── agreement_analysis.py
│   └── efficiency_analysis.py
│
├── escalation/
│   ├── run_escalation.py             ← was run_phase2_hitl.py (folder runner)
│   ├── config.py                     ← was escalation_config.py
│   ├── engine.py                     ← was escalation_engine.py
│   ├── rules.py                      ← was escalation_rules.py
│   ├── meta_feature_builder.py       ← was step2_feature_builder.py
│   ├── meta_model_trainer.py         ← was step2_train_meta_model.py
│   ├── meta_model_inference.py       ← was step2_new_patient_meta.py
│   ├── inference_batch.py            ← was run_inference_batch.py
│   ├── inference_single.py           ← was run_inference_single.py
│   ├── threshold_runner.py           ← was run_step3_thresholds.py
│   ├── threshold_optimizer.py        [unchanged]
│   ├── shap_utils.py                 [unchanged]
│   ├── model_stub.py                 [unchanged]
│   └── test_escalation.py            ← was test_step2_escalation.py
│
├── hitl/
│   ├── run_experiment.py             ← was experiment_runner.py (folder runner)
│   ├── feedback_loop.py              [unchanged]
│   ├── simulated_clinician.py        [unchanged]
│   └── interaction_logger.py         [unchanged]
│
├── dashboard/
│   └── app.py                        [unchanged except 2 escalation import lines]
│
└── utils/
    ├── __init__.py                   [unchanged]
    ├── constants.py                  [unchanged]
    ├── data_helpers.py               [unchanged]
    └── project_config.py             [unchanged]
```

---

## Execution Order (with verification after each step)

| Step | Action | Verify |
|---|---|---|
| A | Delete `escalation/temp.py` | — |
| B | Rename `eval/` folder → `model_eval/`; rename `performance_analysis1.py` → `performance_analysis.py` inside it; update `run_evaluation.py` import | `python run_evaluation.py` runs cleanly |
| C | Rename all 10 escalation files; update all bare imports inside escalation/; update `dashboard/app.py` 2 import lines | `python escalation/run_escalation.py --run-testset-demo` runs cleanly |
| D | Rename `hitl/experiment_runner.py` → `hitl/run_experiment.py` | `python hitl/run_experiment.py --accuracies 0.85 0.9 0.95` runs cleanly |
| E | Create `evaluation/run_analysis.py` (new runner) | `python evaluation/run_analysis.py` runs cleanly |
| F | Rename `run_deterministic_pipeline.py` → `run_all.py`; create `run_pipeline.py` (Benjamin's 6 only); rename `run_validation_pipeline.py` → `run_analysis_suite.py`; update all subprocess paths inside both | `python run_all.py` runs cleanly |
| G | Final smoke test | `streamlit run dashboard/app.py` loads without import errors |

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Escalation bare imports — any missed rename silently breaks the pipeline | After Step C: `grep -r "escalation_config\|escalation_engine\|escalation_rules\|step2_feature\|step2_train\|step2_new\|run_inference_batch\|run_inference_single\|run_step3" src/` must return 0 results |
| `run_analysis_suite.py` still calls `escalation/run_step3_thresholds.py` (now `threshold_runner.py`) | Removed from pipeline — only kept as standalone file |
| `performance_analysis.py` (was `performance_analysis1.py`) is not imported anywhere | Safe — `model_eval/__init__.py` explicitly excludes it |
