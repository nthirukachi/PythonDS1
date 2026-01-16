---
marp: true
theme: default
paginate: true
---

# Canary Rollout Prediction Shift Analysis

## Detecting and Responding to ML Model Deployment Issues

**Learning Objective:** Understand how to identify, diagnose, and respond to prediction distribution shifts during canary deployments.

---

# Slide 2: Problem Statement

## The Scenario

- **New classifier model** deployed via canary rollout (10% traffic)
- **After 2 hours:** Class A predictions shift from **20% → 55%**
- **Latency:** Normal ✅
- **Error Rate:** Normal ✅
- **Ground Truth Labels:** NOT available yet ❌

> **Challenge:** The model isn't crashing, but predictions have changed dramatically. What do we do?

---

# Slide 3: Real-World Use Case

## Where This Happens

| Industry | Example |
|----------|---------|
| **Finance** | Fraud detection model suddenly flags 3x more transactions |
| **Healthcare** | Diagnosis model predicts "high risk" more frequently |
| **E-commerce** | Recommendation model favors one product category |

## Why It Matters

- Silent failures = users get wrong predictions
- No errors = hard to detect automatically
- Delayed labels = can't measure accuracy immediately

---

# Slide 4: Input Data Overview

## What Data Do We Have?

| Data Source | Description |
|-------------|-------------|
| **Baseline Predictions** | 10,000 predictions from old model |
| **Canary Predictions** | 10,000 predictions from new model |
| **Baseline Features** | Input data during baseline period |
| **Canary Features** | Input data during canary period |

## Key Features

- `feature_1`: Numerical (mean ~50)
- `feature_2`: Numerical (mean ~100)
- `feature_3`: Uniform (0-1)

---

# Slide 5: Concepts Used (High Level)

## Core Concepts

1. **Canary Rollout** - Gradual deployment strategy
2. **Prediction Drift** - Output distribution changes
3. **Covariate Shift** - Input distribution changes
4. **PSI** - Population Stability Index
5. **KS-Test** - Kolmogorov-Smirnov test
6. **Chi-Square Test** - Categorical distribution comparison

---

# Slide 6: Concepts Breakdown

## Quick Definitions

| Concept | One-Liner |
|---------|-----------|
| **Canary** | Route 10% traffic to new model, 90% to stable |
| **PSI** | Measures distribution shift; >0.25 = action needed |
| **KS-Test** | Compares two distributions; p<0.05 = different |
| **Covariate Shift** | Input data looks different than training data |

## Analogy

> Like a doctor diagnosing patients from a new country — same criteria, different population.

---

# Slide 7: Step-by-Step Solution Flow

## Our Approach

```
Step 1: Simulate baseline and canary data
          ↓
Step 2: Check data quality
          ↓
Step 3: Detect input drift (PSI, KS-test)
          ↓
Step 4: Analyze prediction behavior
          ↓
Step 5: Calculate risk signals
          ↓
Step 6: Recommend safest action
```

---

# Slide 8: Code Logic Summary

## Key Functions

| Function | Purpose |
|----------|---------|
| `simulate_baseline_predictions()` | Generate baseline class distribution |
| `simulate_canary_predictions()` | Generate shifted predictions |
| `check_data_quality()` | Nulls, types, outliers |
| `calculate_psi()` | Population Stability Index |
| `detect_input_drift()` | Run PSI + KS tests |
| `analyze_prediction_behavior()` | Compare class distributions |
| `determine_safest_action()` | Decision logic |

---

# Slide 9: Important Functions & Parameters

## PSI Calculation

```python
def calculate_psi(expected, actual, buckets=10):
    # buckets: Number of bins (default 10)
    # Returns: PSI value (float)
```

## KS Test

```python
from scipy import stats
ks_stat, p_value = stats.ks_2samp(baseline, canary)
# p_value < 0.05 → significant difference
```

---

# Slide 10: Execution Output

## Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| feature_1 PSI | 0.32 | SIGNIFICANT DRIFT |
| Class A Shift | +35% | MAJOR SHIFT |
| Chi-Square p-value | <0.001 | Distributions differ |

## Recommended Action

**PAUSE + ROUTE TO REVIEW** (75% confidence)

---

# Slide 11: Observations & Insights

## What We Learned

1. **Class A predictions nearly tripled** (20% → 55%)
2. **feature_1 drifted** (mean 50 → 65)
3. **No data quality issues** detected
4. **Model running fine technically** (no errors)

## Key Insight

> A model that runs without errors is NOT the same as a model that produces correct predictions.

---

# Slide 12: Advantages & Limitations

## Advantages of This Approach

✅ Detects issues before labels arrive
✅ Uses statistical rigor (not gut feeling)
✅ Provides confidence levels for decisions
✅ Works for any classification model

## Limitations

❌ Cannot measure true accuracy without labels
❌ PSI sensitive to bucket size
❌ May miss multivariate drift

---

# Slide 13: Interview Key Takeaways

## Common Questions

1. **Q:** What causes prediction drift without errors?
   **A:** Input drift, calibration differences, feature mismatch

2. **Q:** When to rollback vs pause?
   **A:** Rollback if multiple high-severity signals; pause if uncertain

3. **Q:** Why not wait for labels?
   **A:** Users may be affected by wrong predictions in the meantime

## Golden Rule

> "When in doubt, rollback. User trust is harder to rebuild than deployment momentum."

---

# Slide 14: Conclusion

## Summary

- **3 Plausible Causes:** Input drift, feature mismatch, calibration difference
- **First Checks:** Data quality → Input drift → Prediction behavior
- **Safest Action:** PAUSE + ROUTE TO REVIEW (75% confidence)

## Key Learning

Proactive monitoring is essential. Use PSI, KS-test, and chi-square to detect issues before ground truth arrives.

---

# Thank You!

## Questions?

**Project:** canary_rollout_prediction_shift
**Date:** 2026-01-16
