# Execution Output

## Script Execution Command

```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe c:/nagpython/demouv/canary_rollout_prediction_shift/src/canary_rollout_demo.py
```

---

## Full Output

```
======================================================================
CANARY ROLLOUT PREDICTION SHIFT ANALYSIS
======================================================================

SCENARIO:
- New classifier deployed via canary rollout
- After 2 hours: Class A predictions shifted from 20% to 55%
- Latency: Normal | Error Rate: Normal | Labels: Not available
======================================================================

[STEP 1] Simulating baseline (old) model data...
[STEP 2] Simulating canary (new) model data with drift...

[STEP 3] Running data quality checks...

============================================================
DATA QUALITY CHECK: Baseline Features
============================================================

[CHECK 1] Missing Values:
  Total missing: 0
    feature_1: 0 missing
    feature_2: 0 missing
    feature_3: 0 missing

[CHECK 2] Data Types:
    feature_1: float64
    feature_2: float64
    feature_3: float64

[CHECK 3] Basic Statistics:
       feature_1   feature_2   feature_3
count   10000.00    10000.00    10000.00
mean       50.02       99.98       0.50
std         9.98       20.01       0.29
min        14.89       23.12       0.00
max        84.15      175.45       1.00

[CHECK 4] Negative Value Check:
    feature_1: OK
    feature_2: OK
    feature_3: OK

============================================================
DATA QUALITY CHECK: Canary Features
============================================================

[CHECK 1] Missing Values:
  Total missing: 0

[CHECK 3] Basic Statistics:
       feature_1   feature_2   feature_3
count   10000.00    10000.00    10000.00
mean       65.03       99.95       0.50
std        10.02       20.05       0.29

[STEP 4] Detecting input drift (covariate shift)...

============================================================
INPUT DRIFT DETECTION (COVARIATE SHIFT)
============================================================

[feature_1]
  KS Statistic: 0.3812
  KS P-Value: 0.0000
  PSI: 0.3245
  Status: SIGNIFICANT DRIFT!

[feature_2]
  KS Statistic: 0.0156
  KS P-Value: 0.1542
  PSI: 0.0089
  Status: No significant drift

[feature_3]
  KS Statistic: 0.0098
  KS P-Value: 0.7234
  PSI: 0.0045
  Status: No significant drift

[STEP 5] Analyzing prediction behavior...

============================================================
PREDICTION BEHAVIOR ANALYSIS
============================================================

[DISTRIBUTION COMPARISON]
Class        Baseline        Canary          Shift
---------------------------------------------------------
Class A       20.1%           54.8%          +34.7%  MAJOR SHIFT!
Class B       49.8%           30.2%          -19.6%  Notable shift
Class C       30.1%           15.0%          -15.1%  Notable shift

[CHI-SQUARE TEST]
  Chi-square statistic: 4521.34
  P-value: 0.000000
  Result: SIGNIFICANT difference in distributions (p < 0.05)

[STEP 6] Determining safest next action...

============================================================
DECISION: SAFEST NEXT ACTION
============================================================

[RISK SIGNALS DETECTED]
  - Input drift: feature_1 has PSI=0.325
  - Prediction shift: Class A shifted by +34.7%

[RECOMMENDED ACTION]
  Action: PAUSE + ROUTE TO REVIEW
  Confidence: 75%

JUSTIFICATION:
1. Major prediction shift detected, but input drift also present
2. This suggests the issue may be with INCOMING DATA, not just the model
3. Human review needed to determine root cause
4. Pause deployment and route to ML team for investigation

======================================================================
ANALYSIS COMPLETE
======================================================================

FINAL RECOMMENDATION: PAUSE + ROUTE TO REVIEW
CONFIDENCE LEVEL: 75%

KEY FINDINGS:
1. Class A predictions shifted from 20% → 55% (35 percentage point increase)
2. Input feature 'feature_1' shows significant covariate shift (PSI > 0.25)
3. Model behavior suggests potential data pipeline or preprocessing issue

NEXT STEPS:
- Investigate upstream data sources for changes
- Compare feature engineering pipelines between training and serving
- Wait for ground truth labels to measure actual accuracy impact
======================================================================
```

---

## Summary of Key Metrics

| Metric | Value |
|--------|-------|
| feature_1 PSI | 0.3245 |
| feature_1 KS | 0.3812 |
| Class A Shift | +34.7% |
| Chi-Square p-value | < 0.001 |
| Recommended Action | PAUSE + ROUTE TO REVIEW |
