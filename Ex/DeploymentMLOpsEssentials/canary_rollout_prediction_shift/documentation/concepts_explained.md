# Concepts Explained: Canary Rollout Prediction Shift Analysis

This document provides **detailed explanations** of every concept used in this project. Each concept is explained using the **WHY, WHAT, WHEN, WHERE, HOW** framework suitable for complete beginners.

---

## Table of Contents

1. [Canary Rollout](#1-canary-rollout)
2. [Prediction Drift](#2-prediction-drift)
3. [Covariate Shift (Input Drift)](#3-covariate-shift-input-drift)
4. [Population Stability Index (PSI)](#4-population-stability-index-psi)
5. [Kolmogorov-Smirnov (KS) Test](#5-kolmogorov-smirnov-ks-test)
6. [Chi-Square Test](#6-chi-square-test)
7. [Model Calibration](#7-model-calibration)
8. [Rollback Strategy](#8-rollback-strategy)
9. [Shadow Mode Deployment](#9-shadow-mode-deployment)
10. [Ground Truth Labels](#10-ground-truth-labels)

---

## 1. Canary Rollout

### 📖 What is it?

A **canary rollout** is a deployment strategy where a new version of a system (in our case, a machine learning model) is gradually released to a small subset of users before being rolled out to the entire population.

### 🎯 Why is it needed?

Imagine you're a chef testing a new recipe:
- You wouldn't serve the new dish to your entire restaurant immediately
- You'd first serve it to a few trusted tables
- If they like it, you expand; if there's a problem, you've only affected a few customers

**Same logic applies to ML models:**
- Deploying to 100% traffic immediately is risky
- If the model has bugs, ALL users are affected
- Canary lets you catch issues early with minimal impact

### ⏰ When should it be used?

- Deploying a new ML model to production
- Updating model weights or architecture
- Changing feature engineering pipelines
- Any change that could affect prediction quality

### 🏢 Where is it used in industry?

| Company | Use Case |
|---------|----------|
| **Netflix** | Testing new recommendation algorithms |
| **Google** | Rolling out search ranking model updates |
| **Uber** | Deploying pricing or ETA prediction models |
| **Facebook** | Testing content moderation classifiers |

### 🔧 How does it work?

```
Traffic Router
     │
     ├── 90% ──→ Baseline Model (stable)
     │
     └── 10% ──→ Canary Model (new)
```

**The process:**
1. Route 10% of traffic to the new model
2. Monitor for 1-2 hours
3. If metrics are healthy, increase to 25%, then 50%, then 100%
4. If issues detected, rollback immediately

### 🔄 How it works internally?

A **load balancer** or **traffic router** is configured with routing rules:
- `90% → baseline-model-v1`
- `10% → canary-model-v2`

The routing can be:
- **Random** - Every request has a 10% chance of going to canary
- **User-based** - Specific user IDs are always routed to canary
- **Region-based** - A specific geographic region gets canary

### ✅ Advantages

- Low risk: Only small % of users affected by issues
- Fast feedback: Issues detected within hours, not days
- Easy rollback: Just change routing rules back

### ❌ Disadvantages

- Adds complexity to deployment infrastructure
- Requires robust monitoring and alerting
- May mask issues that only appear at full scale

---

## 2. Prediction Drift

### 📖 What is it?

**Prediction drift** (also called **concept drift** or **output drift**) occurs when the distribution of model predictions changes over time, even if the model itself hasn't changed.

### 🎯 Why is it needed to detect?

**Real-life analogy:**
Imagine a doctor who used to diagnose 20% of patients as "high risk." Suddenly, they start diagnosing 55% as "high risk" — but nothing about the patients has visibly changed.

This is alarming! Something must have changed:
- The doctor's judgment?
- The patient population?
- The diagnostic tools?

**For ML models:**
- Prediction drift signals that something has changed
- It could mean model degradation or data issues
- Early detection prevents user-facing problems

### ⏰ When should it be monitored?

- Continuously in production
- Especially after deployments
- During known seasonal changes (holidays, events)
- When upstream data sources change

### 🏢 Where is it used?

| System | Prediction Drift Example |
|--------|--------------------------|
| **Fraud Detection** | Fraud predictions spike from 1% to 5% |
| **Loan Approval** | Rejection rate drops from 30% to 10% |
| **Spam Filter** | Spam classification rises from 20% to 60% |

### 🔧 How to detect?

1. **Track class distribution over time:**
   ```
   Window 1: Class A=20%, Class B=50%, Class C=30%
   Window 2: Class A=55%, Class B=30%, Class C=15%  ← DRIFT!
   ```

2. **Use statistical tests:**
   - Chi-square test for categorical distributions
   - Jensen-Shannon divergence for probability distributions

3. **Set alert thresholds:**
   - Alert if any class changes by more than 10%
   - Alert if chi-square p-value < 0.05

### 🔄 How it works internally?

Models are deterministic: same input → same output.

If predictions change, it must be due to:
1. **Input drift** - The data coming in has changed
2. **Model drift** - The model itself has changed (weights, code)
3. **Feedback loops** - Model predictions affect future data

---

## 3. Covariate Shift (Input Drift)

### 📖 What is it?

**Covariate shift** occurs when the distribution of input features changes between training and production, while the relationship between inputs and outputs (P(Y|X)) remains the same.

### 🎯 Why is it a problem?

**Real-life analogy:**
A teacher learns to grade exams from students aged 15-18. Suddenly, they must grade exams from 8-year-olds. The grading criteria (relationship) is the same, but the student population (inputs) has changed.

**For ML models:**
- Model was trained on data distribution A
- Production data has distribution B
- Model may perform poorly on distribution B
- Even if it was highly accurate on training data

### ⏰ When does it happen?

| Cause | Example |
|-------|---------|
| **Seasonal changes** | Holiday shopping behavior differs from regular |
| **User base changes** | App expands to new country |
| **Data pipeline bugs** | ETL job starts producing nulls |
| **Upstream system changes** | Sensor calibration changes |

### 🏢 Where is it critical?

- **Healthcare:** Patient demographics change
- **Finance:** Economic conditions shift
- **Marketing:** Campaign targets new audience
- **Manufacturing:** Raw materials from new supplier

### 🔧 How to detect?

1. **Compare feature distributions:**
   ```python
   Training: feature_1 mean = 50, std = 10
   Production: feature_1 mean = 65, std = 10  ← SHIFTED!
   ```

2. **Use statistical tests:**
   - KS-test for continuous features
   - Chi-square for categorical features

3. **Calculate PSI (Population Stability Index)**

### 🔄 Visual Example

```
Training Data:               Production Data:
      ____                         ____
     /    \                       /    \
____/      \____           _____/      \____
   40  50  60                   55  65  75

    Mean = 50                    Mean = 65  ← SHIFTED!
```

---

## 4. Population Stability Index (PSI)

### 📖 What is it?

**PSI (Population Stability Index)** is a metric that measures how much a variable's distribution has changed between two time periods or populations.

### 🎯 Why is it needed?

**Real-life analogy:**
Imagine comparing your class photo from 2020 to 2024:
- Same number of students?
- Same distribution of heights?
- Same mix of genders?

PSI quantifies "how different" these photos are.

**For ML:**
- Compare training data to production data
- Quantify drift severity
- Make data-driven rollback decisions

### 📊 PSI Interpretation

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Continue |
| 0.1 - 0.25 | Moderate change | Investigate |
| > 0.25 | Significant change | Action required! |

### 🔧 How to calculate?

**Formula:**
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

**Step-by-step:**
1. Divide data into buckets (e.g., 10 bins)
2. Calculate percentage of data in each bucket for both distributions
3. Apply the formula for each bucket
4. Sum all bucket PSI values

### 💻 Python Example

```python
def calculate_psi(expected, actual, buckets=10):
    # Create buckets based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Calculate frequencies
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Convert to proportions
    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)
    
    # Calculate PSI
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    
    return psi
```

### ✅ Advantages

- Easy to interpret (clear thresholds)
- Works for any numerical variable
- Industry-standard metric

### ❌ Disadvantages

- Sensitive to bucket size
- May miss subtle distribution changes
- Doesn't capture multivariate relationships

---

## 5. Kolmogorov-Smirnov (KS) Test

### 📖 What is it?

The **Kolmogorov-Smirnov (KS) test** is a statistical test that compares two probability distributions to determine if they are different.

### 🎯 Why is it needed?

**Real-life analogy:**
You have two bags of apples. You want to know if they came from the same orchard (same distribution of sizes) or different orchards.

**For ML:**
- Compare training vs production feature distributions
- Detect if incoming data is "from a different orchard"
- Get a statistical p-value for decision-making

### 📊 KS Test Output

| Metric | Description |
|--------|-------------|
| **KS Statistic** | Maximum distance between cumulative distributions (0-1) |
| **P-Value** | Probability distributions are the same |

**Interpretation:**
- P-Value < 0.05 → Distributions are significantly different
- P-Value >= 0.05 → No significant difference detected

### 🔧 How to use?

```python
from scipy import stats

# Compare two distributions
ks_statistic, p_value = stats.ks_2samp(training_data, production_data)

if p_value < 0.05:
    print("Significant difference detected!")
else:
    print("Distributions appear similar")
```

### 🔄 How it works internally?

1. Build cumulative distribution function (CDF) for both samples
2. Find the maximum vertical distance between the two CDFs
3. Calculate p-value based on sample sizes and maximum distance

```
CDF
1.0|     ____----
   |    /   ____----
   |   /___/           ← Maximum distance = KS Statistic
   |  /
0.0|_/________________
      Feature Value
```

### ✅ Advantages

- Non-parametric (no assumptions about distribution shape)
- Provides both statistic and p-value
- Well-established statistical method

### ❌ Disadvantages

- Sensitive to sample size (large samples detect tiny differences)
- Only compares univariate distributions
- P-value alone may not indicate practical significance

---

## 6. Chi-Square Test

### 📖 What is it?

The **chi-square test** is a statistical test that compares observed frequencies with expected frequencies to determine if there's a significant difference.

### 🎯 Why is it needed?

**Real-life analogy:**
You roll a die 600 times. You expect each number to appear ~100 times. If you get:
- 1: 95 times
- 2: 102 times
- 3: 98 times
- 4: 150 times ← This seems off!
- 5: 75 times
- 6: 80 times

Chi-square tells you if this deviation is significant or just chance.

**For ML:**
- Compare baseline vs canary prediction distributions
- Determine if prediction shift is statistically significant
- Supports rollback/continue decisions

### 🔧 How to use?

```python
from scipy import stats

# Expected frequencies (from baseline)
expected = [2000, 5000, 3000]  # Class A, B, C

# Observed frequencies (from canary)
observed = [5500, 3000, 1500]  # Class A, B, C

chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)

if p_value < 0.05:
    print("Prediction distribution has changed significantly!")
```

### 📊 Interpretation

| P-Value | Interpretation |
|---------|---------------|
| < 0.001 | Highly significant difference |
| < 0.05 | Significant difference |
| >= 0.05 | No significant difference |

### ✅ Advantages

- Simple to compute and interpret
- Works for categorical data (prediction classes)
- Widely understood metric

### ❌ Disadvantages

- Requires decent sample sizes (expected > 5 per category)
- Doesn't tell you WHICH categories differ
- Can be overly sensitive with large samples

---

## 7. Model Calibration

### 📖 What is it?

**Model calibration** refers to how well the predicted probabilities of a model match the actual outcomes. A well-calibrated model's predicted probabilities reflect true likelihoods.

### 🎯 Why is it important?

**Real-life analogy:**
A weather forecaster says "70% chance of rain" for 10 different days. If the model is calibrated:
- On days with 70% predictions, it should rain ~7 out of 10 times

If it only rains 3 out of 10 times, the model is **overconfident** and poorly calibrated.

**For ML:**
- Affects threshold-based decisions
- Critical for business cost calculations
- Poor calibration can cause prediction drift even without data drift

### 🔧 How calibration affects predictions

Two models with same accuracy but different calibration:

| Sample | Model A (Calibrated) | Model B (Overconfident) | True Label |
|--------|---------------------|------------------------|------------|
| 1 | 0.70 → Class A | 0.95 → Class A | Class A ✓ |
| 2 | 0.55 → Class A | 0.85 → Class A | Class B ✗ |
| 3 | 0.45 → Class B | 0.80 → Class A | Class B ✓ (Model A) ✗ (Model B) |

If the new (canary) model has different calibration than baseline, prediction distributions will shift!

### 🔄 How to check calibration?

1. **Calibration curve:** Plot predicted probability vs actual frequency
2. **Brier score:** Lower is better (0 = perfect)
3. **Expected Calibration Error (ECE):** Measures average calibration gap

---

## 8. Rollback Strategy

### 📖 What is it?

**Rollback** is the process of reverting a deployment to a previous stable version when issues are detected.

### 🎯 Why is it critical?

**Real-life analogy:**
You're driving a new car and the brakes feel off. You don't keep driving to "test it more." You pull over immediately and get the old car.

**For ML:**
- Protects users from bad predictions
- Preserves trust and business value
- Provides time for root cause analysis

### ⏰ When to rollback?

| Signal | Severity | Action |
|--------|----------|--------|
| Prediction drift + No input drift | Medium | Pause + Review |
| Input drift + Prediction drift | High | Rollback |
| Error rate spike | Critical | Immediate rollback |
| Multiple alerts | High | Rollback |

### 🔧 How to rollback?

```bash
# Update traffic routing
kubectl set routing baseline-model 100% canary-model 0%

# Or in load balancer config
route_percentages:
  baseline-v1: 100%
  canary-v2: 0%     # Disabled
```

### ✅ Key Principle

> "When in doubt, rollback. User trust is harder to rebuild than deployment momentum."

---

## 9. Shadow Mode Deployment

### 📖 What is it?

**Shadow mode** (also called **dark launch**) is a deployment strategy where the new model receives production traffic and makes predictions, but those predictions are NOT served to users. Instead, they're logged for comparison.

### 🎯 Why is it useful?

- Test model on real traffic without user impact
- Compare shadow predictions to baseline predictions
- Build confidence before canary rollout

### 🔧 How it works?

```
User Request
     │
     ├──────→ Baseline Model ──→ Response to User
     │
     └──────→ Shadow Model ──→ Log predictions only (not served)
```

### ✅ When to use?

- Before canary deployment (validate model first)
- When switching to completely new model architecture
- High-risk applications (healthcare, finance)

---

## 10. Ground Truth Labels

### 📖 What is it?

**Ground truth labels** are the actual, correct answers for predictions. They represent reality — what SHOULD have been predicted.

### 🎯 Why is delayed ground truth a problem?

In many ML systems, labels arrive AFTER predictions:
- **Fraud detection:** You know if a transaction was fraud only after investigation (days/weeks)
- **Medical diagnosis:** You know the true diagnosis after tests (days)
- **Content moderation:** You know if content was actually harmful after review

### ⚠️ The Challenge

Without ground truth:
- We CANNOT measure accuracy
- We CANNOT measure precision/recall
- We CAN ONLY observe prediction distributions and input data

This is why statistical tests (PSI, KS, chi-square) are essential — they're the best we can do before labels arrive.

---

## 💡 Key Interview Takeaways

| Concept | One-Liner for Interviews |
|---------|--------------------------|
| **Canary Rollout** | "Gradual deployment routing small % of traffic to new model for early issue detection" |
| **Prediction Drift** | "Model predictions change over time, signaling potential issues" |
| **Covariate Shift** | "Production data distribution differs from training data" |
| **PSI** | "Quantifies distribution shift; >0.25 means significant change" |
| **KS-Test** | "Statistical test comparing two distributions; p<0.05 means they differ" |
| **Rollback** | "Revert to stable version when issues detected; user safety first" |

---

## 📚 Further Reading

1. **Sculley et al. (2015)** - "Hidden Technical Debt in Machine Learning Systems"
2. **Google ML Testing Guide** - "Testing and Monitoring Machine Learning Model Deployments"
3. **Martin Fowler** - "Continuous Delivery for Machine Learning"
