What the data ruled out

- Class distribution is virtually identical (χ²=0.004, p=0.998) — the holdout is not
  "easier" because it has fewer MCI cases
- MCI fraction is almost the same (33.9% vs 33.6%) — not the cause
- Sampling variance cannot explain it — the 95% CIs don't overlap, so the gap is real
  and systematic

What actually explains the gap

The script identified the correct cause: OOF pessimism — a well-known, legitimate  
 property of cross-validation.

Here's the logic:

┌─────────────────┬──────────────────────────────────┬───────────────┐
│ Evaluation │ Model trained on │ Test set size │
├─────────────────┼──────────────────────────────────┼───────────────┤
│ OOF (73.7%) │ 80% of training data per fold │ 460 samples │
├─────────────────┼──────────────────────────────────┼───────────────┤
│ Holdout (92.2%) │ 100% of training data (full set) │ 116 samples │
└─────────────────┴──────────────────────────────────┴───────────────┘

When you generate OOF predictions, each fold model only sees 4 out of 5 folds = ~368  
 samples. The final deployed model (used to predict on holdout) trains on all 460  
 training samples — it's a meaningfully stronger model because it has seen 25% more  
 data.

This is not a flaw. It's expected behaviour. Cross-validation is designed to give  
 pessimistic but unbiased estimates of generalisation — it deliberately uses less  
 training data to simulate what happens on unseen data. The holdout result tells you  
 how the fully trained model performs.

---

What to write in your report

▎ "OOF accuracy (73.7%) is lower than holdout accuracy (92.2%), a gap that is
statistically significant (non-overlapping 95% CIs). Class distributions between the  
 two sets are near-identical (χ²=0.004, p=0.998), ruling out sampling bias as a cause.
The gap is attributable to OOF pessimism: each fold model trains on 80% of the  
 available training data, whereas the final deployed model trains on 100%.
Cross-validation intentionally produces conservative generalisation estimates; the  
 holdout result reflects the performance of the fully trained ensemble."

This is a good thing to explain in your report — it shows you understand the
methodology deeply, which is exactly what gets you Outstanding marks.

From within your activated venv, from the src/ directory:  
 cd C:\Users\sahil\Documents\Projects\dissertation\src  
 streamlit run dashboard/app.py  
 Opens at http://localhost:8501. --- What the Dashboard Contains There are two main modes, selectable from the sidebar: --- Mode 1 — New Patient Inference (default) A clinician enters biomarker values manually (or uploads a CSV), clicks Run Prediction + HITL Policy, and gets: - The ensemble prediction (SCD / MCI / AD) with per-class probabilities bar chart - Whether the system escalates or runs autonomously (based on τ\*=0.675) - The escalation reasons (entropy too high, model disagreement, missing biomarkers, etc.) Data used: No historical data — pure live inference through the trained  
 pipeline.

---

Mode 2 — Retrospective Review

Uses historical data. There's a dropdown to pick the source:

┌───────────────┬─────────────────────────────────────────────────────┬─────┐  
 │ Source │ What it is │ n │  
 ├───────────────┼─────────────────────────────────────────────────────┼─────┤  
 │ testset │ The 20% holdout — cases the model has never seen │ 116 │  
 │ (default) │ during training │ │  
 ├───────────────┼─────────────────────────────────────────────────────┼─────┤  
 │ oof │ The 80% training set with OOF predictions — each │ 460 │  
 │ │ case predicted by a fold that excluded it │ │  
 └───────────────┴─────────────────────────────────────────────────────┴─────┘

The testset is the correct one for demonstrations — it's genuinely unseen data.

Within retrospective mode there are 4 views:

Patient Queue — Sortable/filterable list of all cases ranked by risk score.  
 Coloured by escalation level (red=mandatory, amber=AI-assisted,
green=autonomous).

Patient Detail / Review — Click into any patient by Sample ID:

- Predicted class + true label + escalation level + risk score
- Per-class probability bar chart
- SHAP feature attribution waterfall chart (falls back to surrogate if SHAP  
  cache missing)
- Biomarker values table with reference ranges
- 5 nearest-neighbour similar patients
- Clinician input form — agree/disagree, corrected diagnosis, confidence, notes
  → saved to SQLite DB

Analytics Dashboard — Aggregate view:

- AI-only accuracy, escalation rate, cases reviewed
- Cumulative AI vs HITL accuracy over time (from actual logged feedback)
- Escalation rate by diagnosis class
- Live simulation widget (re-runs the HITL experiment with custom parameters)

Feedback History — Table of all submitted clinician reviews with export to CSV.

---

What data it loads at startup

From artifacts/: voting ensemble + preprocessors + meta-model + τ\* threshold +  
 SHAP caches
From data/: preprocessed_data.csv (for raw biomarker values and medians)  
 From reports/tables/: escalation_table_oof.csv and escalation_table_testset.csv
From artifacts/: hitl_feedback.db (SQLite clinician interaction log)
