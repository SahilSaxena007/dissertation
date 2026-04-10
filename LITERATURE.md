## Human Accuracy Assumption for Step 3 (MASTERPLAN 0.4)

### Scope
- Purpose: set a clinically plausible range for simulated clinician accuracy in HITL threshold optimization.
- Applied assumption in code: sensitivity sweep over `human_accuracy in [0.85, 0.95]` (uniform grid, 0.01 step).

### Sources
1. Knopman DS, et al. "Practice parameter: diagnosis of dementia (an evidence-based review)." *Neurology* (2001).
- Link: https://pubmed.ncbi.nlm.nih.gov/11591898/
- Relevance: guideline-style evidence review reporting variability in clinical diagnosis performance of dementia/AD syndromes; supports modeling clinician performance as uncertain rather than fixed.

2. Ritchie K, et al. "The neuropsychological diagnosis of Alzheimer's disease and related dementias: a meta-analysis..." *JAMA* (2007).
- Link: https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/414190
- Relevance: meta-analytic evidence that diagnostic/classification performance varies by setting and method, supporting sensitivity analysis over a range.

3. Lin JS, et al. "Screening for Cognitive Impairment in Older Adults: Updated Evidence Report..." (AHRQ/USPSTF evidence synthesis, 2013 update).
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC4325860/
- Relevance: evidence synthesis showing substantial test/clinical-performance heterogeneity in cognitive impairment workflows; supports using a range rather than a point estimate.

### Implementation Mapping
- Used for `src/escalation/run_step3_thresholds.py` and `src/escalation/threshold_optimizer.py`.
- Step 3 now computes:
  - per-scenario threshold curves (`threshold_analysis_by_human_accuracy.csv`)
  - per-scenario best thresholds (`threshold_sensitivity_summary.csv`)
  - distribution-expected threshold curve and robust chosen threshold (`threshold_analysis.csv`, `step3_best_threshold.json`).

### Access Date
- Accessed: 2026-02-13
