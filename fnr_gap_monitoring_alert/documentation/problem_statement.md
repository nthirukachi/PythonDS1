# 🧩 Problem Statement: FNR Gap Monitoring Alert System

## 📌 What Problem Are We Solving?

In deployed machine learning systems (like medical diagnosis or fraud detection), we face a critical challenge: **unequal harm across different groups**. 

Imagine a hospital uses an AI system to detect diseases. The system might work well overall, but what if it misses more cases at Hospital A than at Hospital B? This is called **unequal harm** - different groups experiencing different error rates.

### The Specific Problem

When a binary classifier is deployed in production:
- Labels (actual outcomes) become available weekly
- Different operational slices (groups) may have different error rates
- We need to detect when the False Negative Rate (FNR) varies too much between groups
- Missing positive cases (false negatives) can cause serious harm (e.g., missed disease diagnoses)

---

## 🎯 Why This Matters (Real-World Relevance)

### Healthcare Example
- A diagnostic AI system is deployed across 5 hospitals
- Hospital A serves a different patient population than Hospital B
- If the AI misses more diseases at Hospital A, those patients suffer **unequal harm**
- We need to detect and fix this before it causes serious damage

### Finance Example
- A fraud detection system runs on mobile and desktop devices
- Mobile users might experience more missed fraud cases
- This creates unfair treatment based on device type

### The Cost of Ignoring This
1. **Patient harm** - Missed diagnoses lead to delayed treatment
2. **Legal liability** - Discrimination lawsuits
3. **Reputation damage** - Loss of trust in AI systems
4. **Regulatory penalties** - Violation of fairness requirements

---

## 📐 The Mathematical Solution

### Step 1: Define FNR per Slice

For each operational slice `g` (like Hospital_A or Device_Mobile):

```
FNR_g = FN_g / (TP_g + FN_g)
```

Where:
- `FN_g` = False Negatives for slice g (cases we missed)
- `TP_g` = True Positives for slice g (cases we correctly identified)
- `FNR_g` = False Negative Rate for slice g

### Step 2: Calculate the Gap

```
Gap = max_g(FNR_g) - min_g(FNR_g)
```

This tells us the difference between the **worst** performing slice and the **best** performing slice.

### Step 3: Alert When Gap Exceeds Threshold

```
Alert fires when: Gap > Threshold (e.g., 0.10 or 10%)
```

---

## 🪜 Steps to Solve the Problem

### Step 1: Collect Classification Data
- Gather weekly prediction results
- Include True Positives (TP) and False Negatives (FN) per slice

### Step 2: Calculate FNR per Slice
- For each operational slice, compute FNR
- FNR = FN / (TP + FN)

### Step 3: Use Rolling Time Window
- Aggregate data over 4 weeks (reduces noise)
- More stable metrics than single-week snapshots

### Step 4: Calculate Gap Metric
- Find max FNR across all slices
- Find min FNR across all slices
- Gap = max - min

### Step 5: Check Alert Threshold
- If Gap > 0.10 (10%), fire alert
- Otherwise, continue monitoring

### Step 6: Execute Runbook (if alert fires)
- Immediate actions within 24 hours
- Short-term actions within 1 week
- Follow-up actions within 1 month

---

## 🎯 Expected Output

### When No Alert Fires
```
✅ NO ALERT: FNR Gap Monitoring Report - Week 4
Gap Value:     0.0823 (8.23%)
Threshold:     0.1000 (10.00%)
Status:        WITHIN LIMITS
```

### When Alert Fires
```
🚨 ALERT FIRED: FNR Gap Monitoring Report - Week 6
Gap Value:     0.1245 (12.45%)
Threshold:     0.1000 (10.00%)
Status:        EXCEEDED

⚠️ WORST Hospital_C: 0.2156 (21.56%)
✅ BEST  Hospital_A: 0.0911 (9.11%)
```

### Runbook Execution
```
# RUNBOOK EXECUTION - IMMEDIATE ACTIONS

1. ⏹️  PAUSE DEPLOYMENT (if critical)
2. 📧 NOTIFY STAKEHOLDERS
3. 🔍 INITIAL INVESTIGATION
4. 📊 GATHER DIAGNOSTIC DATA
...
```

---

## 📊 Success Criteria

1. ✅ System calculates FNR correctly for each slice
2. ✅ Gap metric is computed accurately
3. ✅ Alerts fire when gap exceeds threshold
4. ✅ Runbook provides actionable steps
5. ✅ Time window smooths out weekly noise
6. ✅ System handles edge cases (zero samples, etc.)

---

## 🔑 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Metric | FNR (not FPR) | False negatives cause more harm in healthcare |
| Gap formula | max - min | Simple, interpretable disparity measure |
| Threshold | 10% | Industry standard for fairness monitoring |
| Time window | 4 weeks | Balances stability with responsiveness |
| Action trigger | Automatic | Reduces human oversight burden |

---

## 📚 Concepts You Will Learn

1. **False Negative Rate (FNR)** - Measuring missed positives
2. **Operational Slices** - Grouping data for fairness analysis
3. **Gap Metrics** - Quantifying disparity between groups
4. **Rolling Windows** - Smoothing time-series data
5. **Alert Thresholds** - Defining acceptable vs. unacceptable ranges
6. **Runbooks** - Documenting incident response procedures
7. **Responsible AI** - Ensuring fair treatment across groups
