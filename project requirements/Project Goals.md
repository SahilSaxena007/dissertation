
#### 1. Analysis of the provided AI code (performance characteristics and  error patterns) - <font style="color:green">CORE ML</font>
- This is _actual model analysis_: you’d need to understand model internals, feature importance, bias, calibration, confusion matrices, etc.    
- It’s real ML work — especially if you identify systematic misclassifications or dataset biases.    
- Requires statistical and ML knowledge.  
    → **Keep it. This is legitimate ML work.**    
#### 2. Identification of cases where human intervention would be most valuable -<font style="color:#5A54EC">Conversion to ML possible</font>
- If you base this on confidence scores only, that’s trivial thresholding.    
- Unless you build a [[Meta-Model]] to _predict_ where humans add value (research-style), it’s not ML. 
    → **Mostly conceptual, not core ML.**    
#### 3. Design decision thresholds that trigger human review - <font style="color:green">CORE ML</font>
- Involves probability calibration, ROC/PR trade-offs, uncertainty estimation.    
- Even if not “researchy,” it’s solid applied ML that needs understanding of model behavior.    
- Could be extended into a small research angle (e.g., adaptive thresholding).  
    → **Counts as core ML.**

#### 4. Develop  the way to present AI results to human experts - <font style="color:#FA7979">NOT ML - UX/UI</font>

#### 5. Create methods for capturing, validating, and incorporating human feedback - <font style="color:yellow">BORDERLINE ML</font> - <font style="color:#5A54EC">Conversion to ML possible</font>
- If you simply store human feedback → _not ML_.    
- If you use it to retrain, fine-tune, or adapt the model (active learning / [[Reinforcement Learning from Human Feedback]]) → **then yes, real ML**.    
- So: it depends on implementation depth.  
    → **Potentially core ML, if feedback actually influences the model.**

#### 6. Plan for logging all interactions for later analysis - <font style="color:#FA7979">NOT ML - Data Infrastructure</font>

#### 7. Build user interface for clinicians/experts to: - <font style="color:#FA7979">NOT ML - Front-end Engineering</font>
- review AI classifications,
- view relevant features that led to the classification,
- provide corrections or confirmations,
- indicate confidence levels in their decisions  

#### 8. Implement visualization of patient data and AI reasoning: <font style="color:#5A54EC">Conversion to ML possible</font>: ([[Explainable AI-XAI]])
- If you use SHAP/LIME and interpret results quantitatively → low-level ML.    
- If you’re just showing data visually → not ML.  
    → **Counts only if you use model explainability methods systematically.**

#### 9. Connect the HITL interface with the provided AI code - - <font style="color:#FA7979">NOT ML - Integration/dev work.</font>

#### 10. Develop the feedback loop mechanism - <font style="color:yellow">BORDERLINE ML</font> - <font style="color:#5A54EC">Conversion to ML possible</font>: ([[Automatic Feedback Loop]])
- If it’s a loop that automates retraining → partly ML.    
- If it’s just a data pipeline that stores human corrections → mostly software engineering.  
    → **Counts only if model updates automatically.**

#### 11. Perform technical validation and usability testing - <font style="color:#FA7979">NOT ML - HCI/QA, not ML.</font>

#### 12. Design experiments comparing AI-only vs. HITL performance - <font style="color:green">CORE ML</font>:
- Requires hypothesis testing, defining metrics, significance analysis, etc.    
- Standard ML experimentation discipline.  
    → **Valid ML component.**

#### 13. Measure improvements in  accuracy - <font style="color:green">CORE ML</font>:
- Standard quantitative model evaluation.    
- Not research-heavy but essential part of ML work.  
    → **Counts.**

#### 14. Assess efficiency metrics (time spent, number of interactions) - <font style="color:#FA7979">NOT ML - Human-factors study, not ML.</font>

#### 15. Analyze patterns of AI-human agreement/disagreement - <font style="color:green">CORE ML</font>:
- Quantifying disagreement, understanding uncertainty sources, or identifying domain shifts is ML evaluation work.  



