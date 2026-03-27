title: Classification of patients into SCD, MCI and AD

Abstract: As you become old, there is an increasing  chance to get Alzheimer's disease(AD). Once you get this illness it is very difficult to recover from it. 
This illness causes a great burden to the family. 
Before you are diagnozed as AD, you are going through the stage called Mild cognitive impairment (MCI)
and Subjective Cognitive Decline (SCD) which happens 
before MCI.
Please note some litetures describe  SCD  as
subjective memory complaints(SMC), or 
subjective memory impairment (SMI), or 
subjective cognitive complaint (SCC).
Some patients stay in SCD and some convert to MCI which
is unrelated to AD, i.e., non-amnestic MCI
and some convert to MCI due to AD, i.e., amnestic MCI.
There will be a disease modifiable therapy(DMT)
which can be applicable only to patients with SCD
who are going to convert to AD eventually
(trial started in the US, nothing in the UK though).
DMT may be able to
reduce  the speed  of progress of the illness.
So the detection of SCD which shall be  converted to AD eventually
is very important
to select patients for DMT.

Some MCI patients stay at the stage of MCI
and some MCI patients shall convert to AD
and some shall develop dementia which is unrelated to AD.
There are some treatments which can be applicable to patients
with MCI who are going to  convert to AD
( not as effective as DMT though).
So the detection of MCI which shall be converted to AD is also
important.
However all of those prediction can be carried out after
we understand the stage of the patient such as SCD, MCI and AD.


This project builds upon  previous
student's code which has  performed
the classification of patients int NC(normal condition)/SCD,MCI,AD
and add the feature of human-in-the-loop
to the code to achieve the
interaction between computer and clinicians.

You shall be using Benjamin's code:

The work to be done   is as follows:

- Analysis of the provided AI code (performance characteristics and  error patterns)
- Identification of cases where human intervention would be most valuable

- Design decision thresholds that trigger human review
- Develop  the way to present
AI results to human experts
- Create methods for capturing, validating, and incorporating human feedback
-  Plan for logging all interactions for later analysis

- Build user interface for clinicians/experts to
1)review AI classifications,
2)view relevant features that led to the classification,
3)provide corrections or confirmations,
4)indicate confidence levels in their decisions

- Implement visualization of patient data and AI reasoning
- Connect the HITL interface with the provided AI code
- Develop the feedback loop mechanism
- Perform technical validation and usability testing
- Design experiments comparing AI-only vs. HITL performance
- Measure improvements in  accuracy
- Assess efficiency metrics (time spent, number of interactions)
- Analyze patterns of AI-human agreement/disagreement



--------
Initial reading materials are
classificationSCD.pdf
alzheimer1.pdf
alzheimer2.pdf
alzheimer3.pdf
alzheimer4.pdf
at pastwork/DATA/readingmaterial/.

In order to understand the biomarkers
and how to deal with the data
 in ADNI properly
you should fully understand
/pastwork/PlasmaEstfromCS/georgios.pdf
This document  deals with the
 problem which
is different from this projec
t. However
it explains the data in ADNI in a very
detailed manner.

Previous students' theses in the area of AD and the associated code are at
 pastwork/haveyougotAlzheimer/
 pastwork/ConversionAD
  pastwork/ConversionMCI
 pastwork/HippocampusVolEst
 pastwork/PlasmaEstfromCSF