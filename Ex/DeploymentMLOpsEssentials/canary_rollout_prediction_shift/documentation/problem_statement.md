# Problem Statement: Canary Rollout Prediction Shift Analysis

---

## 🧩 Problem Statement

You deploy a new classifier model using a canary rollout. After 2 hours, the predicted class distribution shifts sharply (Class A: 20% → 55%), but latency and error rate look normal and labels are not available yet. Explain: (i) 3 plausible causes, (ii) what checks you would run first (data quality, input drift, prediction behavior), and (iii) the safest next action (continue, pause, rollback, or route to review) with justification.

### Scenario Description

A machine learning team deploys a **new classifier model** using a **canary rollout strategy**. In this deployment approach:
- 10% of production traffic is routed to the new (canary) model
- 90% of traffic continues to use the stable baseline model

### Observations After 2 Hours

After 2 hours of deployment, the following metrics are observed:

| Metric | Baseline Model | Canary Model | Status |
|--------|---------------|--------------|--------|
| **Latency** | 50ms | 52ms | Normal ✅ |
| **Error Rate** | 0.1% | 0.1% | Normal ✅ |
| **Class A Predictions** | 20% | 55% | **SHIFTED!** ⚠️ |
| **Class B Predictions** | 50% | 30% | Changed |
| **Class C Predictions** | 30% | 15% | Changed |
| **Ground Truth Labels** | — | — | Not available ❌ |

### Why This Matters

The prediction distribution has shifted **dramatically**:
- Class A predictions increased from 20% to 55% (a **35 percentage point** increase!)
- This represents a **175% relative increase** in Class A predictions

**But the model isn't throwing errors.** It's:
- Responding with low latency ✅
- Not crashing ✅
- Making predictions confidently ✅

This is called a **"silent failure"** — the model appears healthy but may be producing incorrect results.

### The Challenge

**Without ground truth labels, we cannot directly measure accuracy.**

We must:
1. Identify **plausible causes** for the shift
2. Run **diagnostic checks** to understand root causes
3. Decide the **safest next action** before significant harm occurs

---

## 🪜 Steps to Solve the Problem

### Step 1: Identify Plausible Causes

Brainstorm potential reasons for the prediction shift:

1. **Input Data Drift (Covariate Shift)**
   - The distribution of incoming data has changed
   - New users, seasonal patterns, or upstream pipeline changes
   
2. **Feature Engineering Mismatch**
   - Training preprocessing ≠ Serving preprocessing
   - Features computed differently in production
   
3. **Model Calibration Difference**
   - New model has different decision thresholds
   - Probability calibration differs from baseline

### Step 2: Run Diagnostic Checks

Perform systematic checks in order of priority:

| Check Type | What to Look For | Tools/Methods |
|------------|------------------|---------------|
| **Data Quality** | Missing values, schema changes, outliers | Null counts, data type validation |
| **Input Drift** | Feature distribution changes | PSI, KS-Test, histogram comparison |
| **Prediction Behavior** | Class distribution, confidence scores | Chi-square test, threshold analysis |

### Step 3: Compare Baseline vs Canary

Create side-by-side comparisons:
- Feature distributions (histograms)
- Prediction distributions (bar charts)
- Statistical test results (p-values)

### Step 4: Calculate Risk Signals

Quantify the severity of each finding:
- PSI > 0.25 = **Significant drift**
- Prediction shift > 20% = **Major change**
- Multiple signals = **Compound risk**

### Step 5: Determine Safest Action

Based on risk signals, choose one of:
- **CONTINUE** - All checks pass, proceed with caution
- **PAUSE** - Minor concerns, hold and monitor
- **ROLLBACK** - Significant issues, revert to baseline
- **ROUTE TO REVIEW** - Uncertain, needs human judgment

### Step 6: Document and Communicate

Record findings for:
- Incident review (if rollback)
- Model improvement (root cause analysis)
- Knowledge sharing (prevent future issues)

---

## 🎯 Expected Output (OVERALL)

### Final Deliverables

1. **Diagnostic Report**
   - Data quality check results
   - Input drift analysis (PSI, KS-test values)
   - Prediction behavior comparison

2. **Risk Assessment**
   - List of detected risk signals
   - Severity classification for each signal

3. **Recommended Action**
   - Clear recommendation (continue/pause/rollback/review)
   - Justification with supporting evidence
   - Confidence level in the recommendation

4. **Next Steps**
   - Immediate actions to take
   - Follow-up investigations needed
   - Monitoring improvements suggested

### Sample Expected Output

```
============================================================
FINAL RECOMMENDATION: PAUSE + ROUTE TO REVIEW
CONFIDENCE LEVEL: 75%

KEY FINDINGS:
1. Class A predictions shifted from 20% → 55% (35pp increase)
2. Input feature 'feature_1' shows PSI = 0.32 (>0.25 threshold)
3. No data quality issues detected

JUSTIFICATION:
- Major prediction shift WITH significant input drift
- The model is seeing different data than training
- Without labels, we cannot confirm accuracy
- The safest action is to pause and investigate

IMMEDIATE ACTIONS:
1. Hold canary at current 10% traffic level
2. Alert ML team for urgent investigation
3. Pull shadow predictions for comparison
4. Check upstream data pipeline for changes
============================================================
```

---

## 📚 Real-World Relevance

### Industries Affected by This Scenario

| Industry | Use Case | Risk of Wrong Action |
|----------|----------|---------------------|
| **Finance** | Fraud detection | False negatives = fraud loss; False positives = customer friction |
| **Healthcare** | Diagnosis models | Wrong predictions = patient harm |
| **E-commerce** | Recommendation systems | Bad recommendations = lost revenue |
| **Content Moderation** | Harmful content detection | Missed harmful content = user safety risk |

### Why This is an Interview Favorite

This scenario tests:
1. **Systems thinking** - Understanding ML pipelines end-to-end
2. **Statistical knowledge** - Using proper tests and metrics
3. **Decision under uncertainty** - Making choices without complete information
4. **Risk awareness** - Prioritizing user safety over speed
5. **Communication** - Explaining technical findings to stakeholders

---

## 🔑 Success Criteria

A successful solution should:

| Criteria | Description |
|----------|-------------|
| **Identify 3+ causes** | List plausible reasons for prediction shift |
| **Run diagnostic checks** | Perform data quality, drift, and behavior analysis |
| **Use proper metrics** | Apply PSI, KS-test, chi-square correctly |
| **Make clear recommendation** | Provide actionable next step with justification |
| **Consider business impact** | Balance technical findings with user impact |
| **Document findings** | Create clear, reviewable analysis record |
