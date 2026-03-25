# Final Code Improvements

## Tier 1 — Must Do (High Impact, Reasonable Effort)

### Task 1: Compare 2-3 Meta-Models for `ai_error` Prediction
- **File**: `src/escalation/step2_train_meta_model.py`
- **Current**: Only LogisticRegression used for the meta-model predicting AI errors
- **Improvement**: Compare LR vs RandomForest vs GradientBoosting (or XGBoost) with CV AUC, select best
- **Why**: Professor will ask "Why Logistic Regression? Did you try alternatives?"
- **Effort**: 2-3 hours
- **Status**: [x] Complete

---

### Task 2: Investigate OOF vs Holdout Accuracy Gap
- **Files**: `src/ModelsFinal.py`, `reports/tables/`
- **Current**: OOF accuracy = 73.7%, Holdout accuracy = 92.2% — an 18.5% gap that is suspicious
- **Improvement**: Analyse class distributions, per-class accuracy on both sets, check if holdout is "easy", document findings
- **Why**: A professor will question this gap — if unexplained it undermines the entire evaluation
- **Effort**: 2-3 hours
- **Status**: [x] Complete

---

### Task 3: Compare 2-3 Stacking Meta-Learners
- **File**: `src/ModelsFinal.py`
- **Current**: Stacking Level-1 meta-learner is only LogisticRegression (AUC=0.899 vs averaged ensemble AUC=0.904)
- **Improvement**: Try RF, XGBoost, or SVM as stacking meta-learners, compare systematically, justify final choice
- **Why**: Strengthens ensemble methodology and shows rigour
- **Effort**: 1-2 hours
- **Status**: [ ] Not started

---

## Tier 2 — Should Do (Moderate Impact)

### Task 4: Feature Selection Stability Across CV Folds
- **File**: New analysis script or extend `src/ModelsFinal.py`
- **Current**: SelectKBest with mutual_info_classif picks 12 features, but stability across folds not tested
- **Improvement**: Run SelectKBest within each CV fold, measure Jaccard similarity of selected feature sets, report stability
- **Why**: Shows features are robust, not artefacts of a single split
- **Effort**: 2-3 hours
- **Status**: [ ] Not started

---

### Task 5: Acknowledge Preprocessing Leakage in Report (or Fix)
- **File**: `src/ModelsFinal.py` (fix) or report only (acknowledge)
- **Current**: Imputer, selector, and scaler are fit on the entire training set before OOF generation — minor leakage
- **Option A (Easy)**: Acknowledge in report, argue impact is negligible (RobustScaler, KNNImputer are robust)
- **Option B (Hard)**: Fit preprocessing within each CV fold with per-fold preprocessors
- **Why**: Shows methodological maturity; a knowledgeable examiner may spot this
- **Effort**: 1 hour (acknowledge) or 6-8 hours (fix)
- **Status**: [ ] Not started

---

### Task 6: Extend Human Accuracy Sensitivity Range
- **Files**: `src/escalation/threshold_optimizer.py`, `src/evaluation/cost_analysis.py`
- **Current**: Sensitivity analysis covers human accuracy 0.85–0.95
- **Improvement**: Extend down to 0.70–0.80 to show robustness under pessimistic assumptions
- **Why**: Professor may ask "What if the clinician is less accurate than 85%?"
- **Effort**: 1 hour
- **Status**: [ ] Not started

---

## Tier 3 — Nice to Have (Defer Unless Time Permits)

### Task 7: Fix Preprocessing Within CV Folds
- **File**: `src/ModelsFinal.py`
- **Current**: Preprocessing fitted on full training set before OOF split
- **Improvement**: Move imputer/selector/scaler fitting inside CV loop, store per-fold preprocessors, regenerate all OOF artifacts
- **Why**: Methodologically correct — eliminates train/val leakage in preprocessing
- **Risk**: Regenerating all downstream artifacts (escalation tables, meta-model, thresholds, experiments)
- **Effort**: 6-8 hours
- **Status**: [ ] Not started

---

### Task 8: Add MC Dropout Uncertainty for Neural Network
- **File**: `src/ModelsFinal.py` or new `src/models/mc_dropout.py`
- **Current**: Uncertainty comes from ensemble disagreement and conformal prediction
- **Improvement**: Enable dropout at inference time, run N forward passes, measure prediction variance as epistemic uncertainty
- **Why**: Advanced UQ technique referenced in literature, adds depth to uncertainty quantification
- **Effort**: 4-5 hours
- **Status**: [ ] Not started

---

### Task 9: Intersectional Fairness Analysis
- **File**: `src/evaluation/fairness_analysis.py`
- **Current**: Fairness analysed per single demographic variable (gender, age bins)
- **Improvement**: Cross demographics (e.g., Female + Age 85+), measure compound disparities
- **Why**: More thorough fairness evaluation, shows depth of analysis
- **Effort**: 2-3 hours
- **Status**: [ ] Not started
