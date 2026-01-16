# 📚 Concepts Explained: FNR Gap Monitoring Alert System

This document provides deep explanations of every concept used in the FNR Gap Monitoring Alert System. Each concept follows the complete learning structure: Definition, Why, When, Where, How to Use, How It Works, Visual Summary, Advantages, and Disadvantages.

---

## Table of Contents

1. [False Negative Rate (FNR)](#1-false-negative-rate-fnr)
2. [True Positive (TP)](#2-true-positive-tp)
3. [False Negative (FN)](#3-false-negative-fn)
4. [Operational Slices](#4-operational-slices)
5. [Gap Metric](#5-gap-metric)
6. [Alert Threshold](#6-alert-threshold)
7. [Time Window (Rolling Window)](#7-time-window-rolling-window)
8. [Runbook](#8-runbook)
9. [Confusion Matrix](#9-confusion-matrix)
10. [Binary Classification](#10-binary-classification)
11. [Fairness in Machine Learning](#11-fairness-in-machine-learning)
12. [Unequal Harm](#12-unequal-harm)

---

## 1. False Negative Rate (FNR)

### 📖 Definition
False Negative Rate (FNR) is the proportion of actual positive cases that were incorrectly classified as negative by the model.

**Formula:**
```
FNR = FN / (TP + FN)
```

### 🤔 Why is it used?
FNR tells us how many positive cases we are **missing**. In critical applications like medical diagnosis, missing a positive case (disease) can be life-threatening.

**Real-life analogy:** Imagine a security guard checking bags at an airport. FNR is like the percentage of dangerous items the guard **fails to detect**. A high FNR means the guard is missing too many threats.

### ⏰ When to use it?
- When the **cost of missing a positive case is high**
- Medical diagnosis (missing a disease)
- Fraud detection (missing fraudulent transactions)
- Security screening (missing threats)

### 📍 Where is it used?
| Industry | Application |
|----------|-------------|
| Healthcare | Disease detection, cancer screening |
| Finance | Fraud detection, credit risk |
| Security | Threat detection, spam filtering |
| Manufacturing | Defect detection |

### 🔧 How to use it?

```python
def calculate_fnr(tp, fn):
    """
    Calculate False Negative Rate
    
    Parameters:
    - tp: True Positives (correctly identified positives)
    - fn: False Negatives (missed positives)
    
    Returns:
    - FNR value between 0 and 1
    """
    total = tp + fn
    if total == 0:
        return 0.0
    return fn / total

# Example
tp = 80   # Correctly caught 80 diseases
fn = 20   # Missed 20 diseases
fnr = calculate_fnr(tp, fn)
print(f"FNR = {fnr}")  # Output: FNR = 0.2 (20%)
```

### ⚙️ How it works internally?
1. Count all actual positive cases: `TP + FN`
2. Count how many positives were missed: `FN`
3. Divide missed by total: `FN / (TP + FN)`
4. Result is a ratio between 0 and 1

**Example calculation:**
- Total positive cases = 100
- Correctly identified = 85 (TP)
- Missed = 15 (FN)
- FNR = 15 / 100 = 0.15 (15%)

### 📊 Visual Summary

```
All Actual Positives: ████████████████████ (100)
                      ▲
                      │
              ┌───────┴───────┐
              │               │
        Detected (TP)    Missed (FN)
        ████████████         ████
            85               15
                      │
                      ▼
              FNR = 15/100 = 0.15
```

### ✅ Advantages
- Easy to calculate and interpret
- Directly measures missed positives
- Critical for high-stakes applications

### ❌ Disadvantages
- Ignores false positives
- May not be suitable when false alarms are costly
- Needs ground truth labels

---

## 2. True Positive (TP)

### 📖 Definition
True Positive is when the model correctly predicts a positive case as positive.

### 🤔 Why is it used?
It tells us how many positive cases we correctly identified.

**Real-life analogy:** If a doctor correctly diagnoses 85 patients who actually have a disease, those 85 are True Positives.

### 🔧 How to identify it?

| Actual | Predicted | Result |
|--------|-----------|--------|
| Positive | Positive | **True Positive** ✅ |
| Positive | Negative | False Negative |
| Negative | Positive | False Positive |
| Negative | Negative | True Negative |

### ✅ Advantages
- Direct measure of correct positive detection
- Easy to understand

### ❌ Disadvantages
- Doesn't tell the complete picture alone
- Must be combined with other metrics

---

## 3. False Negative (FN)

### 📖 Definition
False Negative is when the model incorrectly predicts a positive case as negative (a "miss").

### 🤔 Why is it dangerous?
Missing a positive case can have severe consequences:
- Missing a cancer diagnosis → delayed treatment
- Missing fraud → financial loss
- Missing a security threat → potential harm

**Real-life analogy:** A smoke detector that doesn't ring when there's a fire is giving a False Negative.

### 🔧 How to calculate FN?

```python
# FN = Total Actual Positives - True Positives
actual_positives = 100
true_positives = 85
false_negatives = actual_positives - true_positives
print(f"FN = {false_negatives}")  # Output: FN = 15
```

### ✅ Advantages
- Clear indicator of missed cases
- Critical for risk assessment

### ❌ Disadvantages
- Needs ground truth labels
- May vary over time

---

## 4. Operational Slices

### 📖 Definition
Operational slices are subgroups of data based on specific attributes like device type, location, demographic group, or time period.

### 🤔 Why are they important?
A model might perform well **overall** but poorly for **specific groups**. Monitoring slices helps detect hidden disparities.

**Real-life analogy:** A school has an 80% average pass rate. But if you slice by classroom, you might find Class A has 95% and Class B has 65%. This disparity is hidden in the overall average.

### ⏰ When to use slices?
- When fairness across groups matters
- When operational conditions vary (different devices, locations)
- When subpopulations are important

### 📍 Examples of Operational Slices

| Attribute | Example Slices |
|-----------|----------------|
| Device Type | Mobile, Desktop, Tablet |
| Location | Hospital_A, Hospital_B, Hospital_C |
| Demographics | Age group, Region |
| Time | Weekday vs Weekend, Morning vs Night |

### 🔧 How to define slices?

```python
# Define operational slices
operational_slices = [
    'Hospital_A',
    'Hospital_B', 
    'Hospital_C',
    'Device_Mobile',
    'Device_Desktop'
]

# Group data by slice
for slice_name in operational_slices:
    slice_data = data[data['slice'] == slice_name]
    print(f"{slice_name}: {len(slice_data)} records")
```

### ✅ Advantages
- Reveals hidden disparities
- Enables targeted improvements
- Supports fairness monitoring

### ❌ Disadvantages
- More slices = more computation
- Small slices may have noisy metrics
- Requires careful slice definition

---

## 5. Gap Metric

### 📖 Definition
The Gap Metric measures the disparity between the best and worst performing slices.

**Formula:**
```
Gap = max_g(FNR_g) - min_g(FNR_g)
```

### 🤔 Why is it used?
It provides a single number that quantifies **inequality** across groups.

**Real-life analogy:** If the fastest runner in a race finishes in 10 seconds and the slowest in 18 seconds, the "gap" is 8 seconds. This tells you how spread out the performance is.

### ⏰ When to use it?
- When monitoring fairness across groups
- When you need a simple disparity measure
- When comparing performance variability

### 🔧 How to calculate gap?

```python
def calculate_gap(fnr_dict):
    """
    Calculate FNR Gap
    
    Parameters:
    - fnr_dict: Dictionary of slice_name → FNR value
    
    Returns:
    - gap: max(FNR) - min(FNR)
    - worst_slice: name of slice with max FNR
    - best_slice: name of slice with min FNR
    """
    max_fnr = max(fnr_dict.values())
    min_fnr = min(fnr_dict.values())
    
    worst_slice = max(fnr_dict, key=fnr_dict.get)
    best_slice = min(fnr_dict, key=fnr_dict.get)
    
    gap = max_fnr - min_fnr
    
    return gap, worst_slice, best_slice

# Example
fnr_values = {
    'Hospital_A': 0.10,
    'Hospital_B': 0.15,
    'Hospital_C': 0.22,
    'Device_Mobile': 0.12,
    'Device_Desktop': 0.08
}

gap, worst, best = calculate_gap(fnr_values)
print(f"Gap = {gap}")      # Output: Gap = 0.14 (14%)
print(f"Worst = {worst}")  # Output: Worst = Hospital_C
print(f"Best = {best}")    # Output: Best = Device_Desktop
```

### 📊 Visual Summary

```
FNR per Slice:
Device_Desktop  ████                     0.08 ← BEST
Hospital_A      █████                    0.10
Device_Mobile   ██████                   0.12
Hospital_B      ███████                  0.15
Hospital_C      ███████████              0.22 ← WORST
                                              │
                Gap = 0.22 - 0.08 = 0.14 ─────┘
```

### ✅ Advantages
- Simple and interpretable
- Single number for disparity
- Easy to threshold

### ❌ Disadvantages
- Only considers extremes (ignores middle)
- Sensitive to outliers
- Doesn't show distribution shape

---

## 6. Alert Threshold

### 📖 Definition
An alert threshold is a predefined value that, when exceeded, triggers an alert notification.

### 🤔 Why is it used?
Continuous monitoring generates lots of data. Thresholds help us focus on **significant** deviations only.

**Real-life analogy:** A fire alarm has a smoke threshold. It only rings when smoke exceeds a certain level, not for every tiny bit of smoke.

### ⏰ How to choose a threshold?
| Factor | Consideration |
|--------|---------------|
| Risk tolerance | Lower threshold = more alerts, higher safety |
| Historical data | What gap values have occurred before? |
| Business impact | How costly is unequal harm? |
| False alarm cost | Too many alerts = alert fatigue |

### 🔧 How to implement threshold?

```python
ALERT_THRESHOLD = 0.10  # 10%

def check_alert(gap, threshold=ALERT_THRESHOLD):
    """
    Check if gap exceeds threshold
    
    Parameters:
    - gap: Current gap value
    - threshold: Alert threshold (default 10%)
    
    Returns:
    - True if alert should fire, False otherwise
    """
    return gap > threshold

# Example
gap = 0.14
if check_alert(gap):
    print("🚨 ALERT: Gap exceeds threshold!")
else:
    print("✅ OK: Gap within limits")
```

### ✅ Advantages
- Reduces noise
- Focuses attention on important issues
- Enables automation

### ❌ Disadvantages
- Wrong threshold = missed issues or too many alerts
- May need tuning over time
- Binary decision loses nuance

---

## 7. Time Window (Rolling Window)

### 📖 Definition
A rolling window aggregates data over a fixed period of time, moving forward as new data arrives.

### 🤔 Why is it used?
Weekly data can be noisy. A rolling window **smooths** the metrics for more stable monitoring.

**Real-life analogy:** A 7-day moving average of COVID cases smooths out daily fluctuations to show the trend.

### 🔧 How to implement rolling window?

```python
def rolling_window_fnr(data, current_week, window_weeks=4):
    """
    Calculate FNR using rolling window
    
    Parameters:
    - data: DataFrame with weekly data
    - current_week: Current week number
    - window_weeks: Size of rolling window (default 4)
    
    Returns:
    - Aggregated FNR values
    """
    start_week = max(1, current_week - window_weeks + 1)
    end_week = current_week
    
    # Filter to window
    window_data = data[(data['week'] >= start_week) & 
                       (data['week'] <= end_week)]
    
    # Aggregate
    aggregated = window_data.groupby('slice').agg({
        'TP': 'sum',
        'FN': 'sum'
    })
    
    return aggregated

# Example: Week 6 with 4-week window uses weeks 3, 4, 5, 6
```

### 📊 Visual Summary

```
Week:  1   2   3   4   5   6   7   8
       ─   ─   ─   ─   ─   ─   ─   ─
                   │───────────│
                   │  Window   │
                   │ (4 weeks) │
                   └───────────┘
                   Week 4-7 data
                   aggregated
```

### ✅ Advantages
- Reduces noise from single-week fluctuations
- More stable metrics
- Smooths seasonal effects

### ❌ Disadvantages
- Delays detection of sudden changes
- Needs history (can't use immediately)
- May hide recent shifts

---

## 8. Runbook

### 📖 Definition
A runbook is a documented set of procedures to follow when a specific event occurs (like an alert firing).

### 🤔 Why is it used?
When alerts fire, people panic. A runbook provides **clear, step-by-step instructions** so responders know exactly what to do.

**Real-life analogy:** A fire drill procedure is a runbook. When the alarm sounds, everyone knows: stop working, use stairs, gather at meeting point.

### 📍 Structure of a Good Runbook

```
RUNBOOK: FNR Gap Alert Response

1. IMMEDIATE ACTIONS (Within 24 hours)
   - Acknowledge alert
   - Notify stakeholders
   - Begin investigation

2. SHORT-TERM ACTIONS (Within 1 week)
   - Root cause analysis
   - Document findings
   - Propose mitigation

3. FOLLOW-UP ACTIONS (Within 1 month)
   - Implement fix
   - Verify resolution
   - Update processes
```

### ✅ Advantages
- Reduces response time
- Ensures consistent handling
- Supports compliance
- Enables delegation

### ❌ Disadvantages
- Needs maintenance
- May not cover all scenarios
- Can become outdated

---

## 9. Confusion Matrix

### 📖 Definition
A confusion matrix is a table that shows the counts of correct and incorrect predictions.

### 📊 Structure

```
                    Predicted
                  Pos    Neg
               ┌──────┬──────┐
Actual  Pos    │  TP  │  FN  │
               ├──────┼──────┤
        Neg    │  FP  │  TN  │
               └──────┴──────┘

TP = True Positive  (Correct positive)
FN = False Negative (Missed positive)
FP = False Positive (False alarm)
TN = True Negative  (Correct negative)
```

### 🔧 Example

```python
# Example confusion matrix
confusion_matrix = {
    'TP': 85,   # 85 diseases correctly detected
    'FN': 15,   # 15 diseases missed
    'FP': 10,   # 10 false alarms
    'TN': 890   # 890 healthy correctly identified
}

# Calculate FNR
fnr = confusion_matrix['FN'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
print(f"FNR = {fnr}")  # Output: FNR = 0.15
```

---

## 10. Binary Classification

### 📖 Definition
Binary classification is a machine learning task where each sample is assigned to one of two classes (positive or negative).

### 📍 Examples

| Domain | Positive Class | Negative Class |
|--------|----------------|----------------|
| Medical | Disease present | Healthy |
| Fraud | Fraudulent | Legitimate |
| Spam | Spam email | Not spam |
| Security | Threat | Safe |

---

## 11. Fairness in Machine Learning

### 📖 Definition
Fairness in ML means the model performs equitably across different groups, without systematic bias that harms specific populations.

### 🤔 Why does fairness matter?
ML models can perpetuate or amplify existing biases, leading to:
- Discrimination against protected groups
- Legal liability
- Reputational damage
- Reduced trust in AI

### 📊 Types of Fairness

| Type | Definition |
|------|------------|
| **Demographic Parity** | Equal positive prediction rates across groups |
| **Equalized Odds** | Equal TPR and FPR across groups |
| **Equal Opportunity** | Equal FNR across groups |
| **Predictive Parity** | Equal precision across groups |

Our FNR Gap monitoring focuses on **Equal Opportunity** (ensuring equal FNR across slices).

---

## 12. Unequal Harm

### 📖 Definition
Unequal harm occurs when different groups experience different levels of negative outcomes from a model's errors.

### 🤔 Why monitor for it?
Even if overall metrics look good, specific groups may suffer:
- Higher rates of missed diagnoses
- More false arrests
- Unfair loan denials

### 📊 Example of Hidden Unequal Harm

```
Overall FNR:     12% (looks acceptable)

Slice Breakdown:
├── Hospital_A:  8%  (acceptable)
├── Hospital_B:  10% (acceptable)
├── Hospital_C:  18% (TOO HIGH!)
└── Hospital_D:  12% (acceptable)

Gap = 18% - 8% = 10% → ALERT!
```

The overall metric hides the disparity affecting Hospital_C patients.

### ✅ How to address it
1. Monitor slices separately
2. Set gap thresholds
3. Investigate root causes
4. Implement targeted fixes

---

## 📚 Summary Table

| Concept | Formula / Definition | Purpose |
|---------|---------------------|---------|
| FNR | FN / (TP + FN) | Measures missed positives |
| Gap | max(FNR) - min(FNR) | Measures disparity |
| Threshold | Gap > 0.10 | Triggers alert |
| Time Window | 4-week rolling | Smooths noise |
| Runbook | Step-by-step procedure | Guides response |
| Slice | Data subgroup | Enables fairness analysis |
