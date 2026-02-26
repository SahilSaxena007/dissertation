# MASTER PLAN: Achieving 95% in HITL Alzheimer's Classification

## Assessment Weight Breakdown
- **Report: 55%** (Abstract 15%, Background 25%, Technical Quality 35%, Conclusions 15%, Presentation 10%)
- **Achievements: 30%** (Complexity 33%, Scale 33%, Achievement 34%)
- **Screencast: 15%** (4 criteria at 25% each)

> For 95%, you need "Outstanding" (90-100) in nearly every criterion.
> "Outstanding" in Technical Quality = "Very little to fault. Extremely thorough understanding. Exceptional explanations."
> "Outstanding" in Achievement = "Solution is hard to improve upon."

---

## PHASE 0: Fix Fundamental Flaws [Week 1] - CRITICAL
These are methodological errors that would immediately lower your grade.

### 0.1 Fix Data Splitting Strategy
**Problem**: Single 80/20 split + meta-model trained on same test set = data leakage.
**Solution**: Implement nested evaluation strategy:
```
Full dataset
  ├── Outer fold (k=5): for base model evaluation
  │     ├── Train (80%): train base models
  │     └── Test (20%): evaluate base models
  └── Inner split of test predictions:
        ├── Meta-train (60% of test preds): train meta-model
        └── Meta-test (40% of test preds): evaluate HITL system
```
**Alternative** (simpler, given small dataset): Use stratified k-fold cross-validation (k=5 or k=10) for the base models, collect out-of-fold predictions, then use THOSE for the meta-model with its own cross-validation.

### 0.2 Implement Proper Cross-Validation
- Replace single train/test split with Stratified K-Fold (k=5 minimum, k=10 preferred given small data)
- Report mean +/- std across folds for ALL metrics
- Use RepeatedStratifiedKFold for even more robust estimates

### 0.3 Complete Stub Functions
- `compute_shap_instability()`: Implement using variance of SHAP values across models
- `compute_multimodal_mismatch()`: Implement biomarker vs cognitive score conflict detection
- `data_helpers.py`: Either implement or remove

### 0.4 Fix human_accuracy Assumption
- Model human accuracy as a distribution (e.g., 0.85-0.95) based on literature
- Run sensitivity analysis: how does optimal threshold change with human accuracy?
- Cite literature on clinician diagnostic accuracy for AD stages

### 0.5 Address Small Dataset Problem
**Options (implement at least 2)**:
1. Use out-of-fold predictions from k-fold CV (no data wasted)
2. Stratified bootstrapping for the meta-model evaluation
3. Generate calibrated synthetic patients using SMOTE/ADASYN specifically for meta-model evaluation
4. Use Leave-One-Out CV for the meta-model if data is very small

---

## PHASE 1: Enhance ML Core [Weeks 1-2] - HIGH IMPACT ON COMPLEXITY

### 1.1 Probability Calibration (ESSENTIAL for escalation)
The escalation engine depends on well-calibrated probabilities. Without calibration, your thresholds are arbitrary.
- Implement Platt scaling (sigmoid calibration)
- Implement isotonic regression calibration
- Compare calibrated vs uncalibrated with reliability diagrams
- Use CalibratedClassifierCV from sklearn
- **Report**: Show calibration curves before/after, compute Expected Calibration Error (ECE)

### 1.2 Expanded Model Zoo
Add at least 2 m/ore base learners to strengthen the ensemble:
- **XGBoost**: Strong gradient boosting alternative to CatBoost
- **LightGBM**: Fast, handles missing values natively
- **SVM with RBF kernel**: Different inductive bias (margin-based)
- **Logistic Regression (L1/L2)**: Linear baseline (important for interpretability comparison)
Compare all models systematically. This shows breadth and rigor.

### 1.3 Stacked Generalization (Replace Simple Averaging)
Current: Simple mean of 3 models' probabilities
Improved: Train a meta-learner (Level-1) on base model outputs
- Use out-of-fold predictions as Level-1 features
- Level-1 model: Logistic Regression or small neural network
- This is a genuine ML contribution, not just averaging

### 1.4 Uncertainty Quantification (ESSENTIAL for HITL)
Go beyond simple entropy/margin:
- **MC Dropout**: Add dropout to NN, run N forward passes, measure prediction variance
- **Ensemble Disagreement**: Already have this, but formalize it (Jensen-Shannon divergence between model distributions)
- **Conformal Prediction**: Provide prediction sets with coverage guarantees
- **Epistemic vs Aleatoric Uncertainty**: Separate model uncertainty from data noise
This is a major complexity differentiator. Conformal prediction especially is cutting-edge and directly relevant to clinical decision-making.

### 1.5 Feature Importance & Selection Robustness
- Run stability analysis on SelectKBest: how stable are the 12 features across CV folds?
- Compare with LASSO-based selection, mutual information, and recursive feature elimination
- Plot feature importance consistency across methods
- This addresses "are the right features being used?" which is critical for clinical trust

---

## PHASE 2: Build the HITL System [Weeks 2-4] - HIGH IMPACT ON ACHIEVEMENT

### 2.1 Clinician Dashboard (Streamlit)
Build a Streamlit web application with these views:

**View 1: Patient Queue**
- List of patients pending review (sorted by risk score)
- Color-coded escalation levels (green/amber/red)
- Filter by escalation level, confidence, class

**View 2: Patient Detail / Review Page**
- AI classification with confidence bars for each class
- SHAP waterfall plot showing feature contributions
- Patient biomarker values with reference ranges
- Comparison to similar patients (nearest neighbors)
- Escalation reason breakdown
- **Clinician input form**: Agree/Disagree, Correct diagnosis, Confidence (1-5 scale), Free-text notes

**View 3: Analytics Dashboard**
- AI-only vs HITL accuracy over time
- Agreement/disagreement patterns
- Escalation rate by class
- Clinician confidence distributions
- Model performance by demographic

**View 4: Feedback History**
- Log of all clinician interactions
- Export capability for analysis

### 2.2 Feedback Capture & Validation
- Store all clinician decisions in structured format (SQLite or CSV)
- Validate inputs (confidence must be 1-5, diagnosis must be valid class)
- Track session metadata (timestamp, time-to-decision, features viewed)

### 2.3 Feedback Loop Mechanism (CORE ML COMPONENT)
This is what transforms the project from "classification + escalation" to genuine HITL:

**Option A: Active Learning Loop** (Recommended)
1. Clinician corrects a misclassified patient
2. Corrected sample is added to a "feedback pool"
3. After N corrections, retrain the model on original training + feedback pool
4. Re-evaluate: did accuracy improve?
5. Simulate this process to show the learning curve

**Option B: Online Model Update**
- Implement incremental learning (partial_fit for compatible models)
- Track model performance before/after each batch of human corrections

**Option C: Confidence-Weighted Retraining**
- Weight human-corrected samples by clinician confidence
- Higher confidence corrections have more influence on retraining
- This is a novel contribution if implemented well

### 2.4 Simulated Clinician Experiment
Since you don't have real clinicians yet:
- Create a "simulated clinician" with configurable accuracy (e.g., 85%, 90%, 95%)
- Run the full HITL pipeline with simulated feedback
- Compare AI-only vs HITL at different clinician accuracy levels
- This is methodologically sound and publishable
- Can be validated later with real clinicians (good "further work")

### 2.5 Interaction Logging
- Log every classification, escalation decision, and clinician interaction
- Store with timestamps for temporal analysis
- Track: time-to-decision, features examined, confidence levels
- This supports the "efficiency metrics" requirement

---

## PHASE 3: Experimental Evaluation [Weeks 3-5] - HIGH IMPACT ON TECHNICAL QUALITY

### 3.1 AI-Only vs HITL Comparison Experiment
**Experimental Design:**
1. **Baseline**: AI-only (ensemble) accuracy on test set
2. **HITL-Oracle**: HITL with perfect human (upper bound)
3. **HITL-Realistic**: HITL with simulated clinician (85-95% accuracy)
4. **HITL-Active**: HITL with active learning after N feedback rounds
5. **Human-Only**: Simulated clinician accuracy (lower bound for comparison)

**Metrics for each condition:**
- Accuracy, F1, AUC (per-class and macro)
- Sensitivity and specificity for each stage
- Expected Calibration Error
- Cost-weighted accuracy (misclassifying AD as SCD is worse than vice versa)

**Statistical Testing:**
- McNemar's test for paired accuracy comparison
- DeLong test for AUC comparison
- Bootstrap confidence intervals for all differences
- Effect size (Cohen's d or odds ratio)

### 3.2 Threshold Sensitivity Analysis
- Vary each escalation threshold independently
- Create sensitivity plots: accuracy vs threshold for each signal
- Multi-objective optimization: accuracy vs review rate Pareto frontier
- Show that chosen thresholds are near-optimal

### 3.3 Ablation Study on Escalation Signals
- Remove one signal at a time, measure impact on HITL performance
- This proves each signal adds value (or doesn't)
- Rank signals by importance using this analysis

### 3.4 Cost-Benefit Analysis
- Assign realistic costs to each type of error (e.g., missing AD diagnosis >> false alarm)
- Optimize threshold under cost constraints, not just accuracy
- This is clinically meaningful and adds real-world relevance

### 3.5 Fairness Analysis
- Evaluate HITL performance across demographic groups (age, gender, education)
- Check if escalation disproportionately affects certain groups
- Report demographic parity and equalized odds metrics

---

## PHASE 4: Code Quality & Structure [Ongoing]

### 4.1 Project Restructure
```
dissertation/
├── README.md
├── CLAUDE.md
├── MASTERPLAN.md
├── requirements.txt
├── Makefile (or run_pipeline.py)
├── config/
│   ├── model_config.yaml       # All hyperparameters
│   ├── escalation_config.yaml  # All thresholds
│   └── experiment_config.yaml  # Experiment settings
├── data/
│   ├── raw/                    # Original ADNI CSVs (Study_files → renamed)
│   ├── processed/              # Merged & preprocessed
│   └── splits/                 # CV fold indices
├── src/
│   ├── data/
│   │   ├── merge.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py            # All model training
│   │   ├── ensemble.py         # Stacking/voting logic
│   │   ├── calibration.py      # Probability calibration
│   │   └── model_registry.py   # Model loading utilities
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   │   ├── uncertainty.py
│   │   ├── statistical_tests.py
│   │   ├── explainability.py
│   │   ├── bias_diagnostics.py
│   │   ├── error_taxonomy.py
│   │   └── orchestrator.py
│   ├── escalation/
│   │   ├── engine.py
│   │   ├── rules.py
│   │   ├── meta_model.py
│   │   ├── threshold_optimizer.py
│   │   └── config.py
│   ├── hitl/
│   │   ├── feedback_loop.py    # Active learning + retraining
│   │   ├── simulated_clinician.py
│   │   ├── interaction_logger.py
│   │   └── experiment_runner.py
│   ├── dashboard/
│   │   ├── app.py              # Streamlit main
│   │   ├── pages/
│   │   │   ├── patient_queue.py
│   │   │   ├── patient_detail.py
│   │   │   ├── analytics.py
│   │   │   └── feedback_history.py
│   │   └── components/
│   │       ├── shap_display.py
│   │       ├── confidence_bars.py
│   │       └── biomarker_table.py
│   └── utils/
│       ├── constants.py
│       ├── paths.py
│       ├── data_helpers.py
│       └── reproducibility.py  # Seed setting, logging
├── experiments/
│   ├── run_cross_validation.py
│   ├── run_hitl_experiment.py
│   ├── run_ablation_study.py
│   ├── run_sensitivity_analysis.py
│   └── run_fairness_audit.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── artifacts/                  # Generated models, splits
├── reports/
│   ├── figures/
│   ├── tables/
│   └── generated/
└── notebooks/                  # EDA, prototyping (optional)
```

### 4.2 Testing
- Migrate from standalone asserts to pytest
- Add integration tests (full pipeline runs)
- Add test for reproducibility (same seed → same results)
- Aim for >80% coverage on core modules

### 4.3 Reproducibility
- Set all random seeds centrally
- Log all hyperparameters and configurations
- Save CV fold indices for reproducibility
- Add a `Makefile` or `run_all.py` that reproduces the entire pipeline

---

## PHASE 5: Report Writing [Weeks 4-7]

### Chapter Structure (targeting 12,000-14,000 words)

**Chapter 1: Introduction** (~1,500 words)
- Alzheimer's disease burden, DMT context, clinical staging (SCD→MCI→AD)
- Why automated classification matters
- Why HITL: limitations of pure AI in clinical settings
- Clear aims & objectives (6-8 specific, measurable aims)
- Evaluation strategy overview
- Report structure

**Chapter 2: Background & Literature Review** (~3,000 words)
- 2.1 Alzheimer's disease & biomarkers (cite medical literature)
- 2.2 Machine learning for AD classification (cite recent papers)
- 2.3 Ensemble methods & uncertainty quantification
- 2.4 Human-in-the-Loop systems (general + medical AI specific)
- 2.5 Active learning & feedback loops
- 2.6 Explainable AI for clinical decision support
- 2.7 Summary & positioning of this project

**Chapter 3: Methodology** (~3,500 words)
- 3.1 Data: ADNI dataset description, feature engineering, preprocessing
- 3.2 Base models: Architecture, hyperparameter search, training procedure
- 3.3 Ensemble: Stacking approach with calibration
- 3.4 Evaluation framework: 11-component system
- 3.5 Escalation engine: Rule design, signal computation
- 3.6 Meta-model: Feature engineering, training, threshold optimization
- 3.7 HITL system: Dashboard, feedback loop, active learning
- 3.8 Experimental design: AI-only vs HITL comparison

**Chapter 4: Results & Evaluation** (~3,000 words)
- 4.1 Base model performance (cross-validated)
- 4.2 Ensemble vs individual models
- 4.3 Calibration analysis
- 4.4 Escalation analysis: what gets escalated and why
- 4.5 Meta-model performance (ROC, precision-recall)
- 4.6 AI-only vs HITL experiment results
- 4.7 Active learning curve
- 4.8 Sensitivity & ablation studies
- 4.9 Fairness analysis
- 4.10 Dashboard usability (if supervisor/clinician feedback available)

**Chapter 5: Discussion** (~1,500 words)
- Interpretation of results
- Clinical implications
- Limitations (honest, specific)
- Comparison with related work

**Chapter 6: Conclusions & Future Work** (~1,000 words)
- Summary of achievements vs aims
- Key contributions
- Future work: real clinician trials, longitudinal tracking, conversion prediction

---

## PHASE 6: Screencast [Week 7-8]
- 7-9 minutes
- Open with Alzheimer's context (emotional hook)
- Show the problem → show the solution
- Live demo of the dashboard
- Key results (accuracy improvement, escalation patterns)
- Theory interwoven (not separate)
- Professional editing, background music

---

## PRIORITY MATRIX (Given 1-2 month timeline)

### Week 1: Fix Foundations + Start ML Enhancements
- [ ] Fix data splitting (cross-validation)
- [ ] Complete stub functions
- [ ] Add probability calibration
- [ ] Add 2 more models (XGBoost, SVM)
- [ ] Implement stacked generalization

### Week 2: HITL Core
- [ ] Build Streamlit dashboard (basic version)
- [ ] Implement feedback capture
- [ ] Implement simulated clinician
- [ ] Implement active learning loop
- [ ] Implement interaction logging

### Week 3: Experiments
- [ ] Run AI-only vs HITL experiment
- [ ] Run ablation study
- [ ] Run sensitivity analysis
- [ ] Run fairness analysis
- [ ] Generate all figures and tables

### Week 4: Polish + Start Report
- [ ] Code cleanup & restructuring
- [ ] Complete testing
- [ ] Begin report writing (Background, Methodology)

### Week 5-6: Report Writing
- [ ] Write full report
- [ ] Generate all figures at publication quality
- [ ] Iterate on narrative

### Week 7: Screencast + Final Polish
- [ ] Record screencast
- [ ] Final report review
- [ ] Submission

---

## WHAT MAKES THIS 95%-WORTHY

### Complexity (Outstanding):
- Combines ML, HCI, uncertainty quantification, active learning, clinical decision support
- Builds on scientific literature (conformal prediction, SHAP, calibration theory)
- Original contribution: meta-model + active learning + threshold optimization under cost constraints

### Scale (Outstanding):
- Multi-model ensemble with stacking
- 11-component evaluation framework
- Full HITL pipeline with dashboard
- Active learning feedback loop
- 5+ formal experiments with statistical testing
- Comprehensive testing and reproducibility

### Achievement (Outstanding):
- ALL professor requirements met
- Solution is grounded in literature
- Convincing experimental evidence
- Dashboard demonstrates real clinical utility
- "Hard to improve upon" for a 3rd year project

---

## THINGS TO LEARN (for your intermediate level)

1. **Probability calibration**: Read sklearn docs on CalibratedClassifierCV + Platt (1999) paper
2. **Stacked generalization**: Read Wolpert (1992) "Stacked Generalization" + sklearn StackingClassifier docs
3. **Conformal prediction**: Read "A Gentle Introduction to Conformal Prediction" (Angelopoulos & Bates 2022)
4. **Active learning**: Read Settles (2009) "Active Learning Literature Survey"
5. **McNemar's test**: Read statsmodels docs + Dietterich (1998) "Approximate Statistical Tests"
6. **SHAP**: You already use it, but read Lundberg & Lee (2017) for the theory
7. **Streamlit**: Official Streamlit docs + gallery examples

---

## USING CLAUDE CODE EFFECTIVELY

### For each phase, use this workflow:
1. **Plan mode** (`/plan` or EnterPlanMode): Before each major task, enter plan mode to design the approach
2. **Subagents**: I'll use explore agents to research patterns and general-purpose agents for complex multi-file changes
3. **Memory**: I'll maintain MEMORY.md with lessons learned across sessions
4. **Iterative development**: Implement → test → evaluate → refine

### Session strategy:
- Each Claude Code session: focus on ONE phase/sub-phase
- Start each session by reviewing MASTERPLAN.md progress
- End each session by updating CLAUDE.md if architecture changed
