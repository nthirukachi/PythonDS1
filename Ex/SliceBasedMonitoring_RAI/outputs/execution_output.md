# 📊 Execution Output: Slice-Based Monitoring Examples

---

## 🔢 Example 1: Loan Approval Model - Overall vs Slice Metrics

### Input Data

```
Total Applications: 10,000
Group A (Majority): 9,000 applications
Group B (Minority): 1,000 applications
```

### Performance Results

```
┌─────────────────────────────────────────────────────────┐
│              LOAN APPROVAL MODEL METRICS                │
├─────────────────────────────────────────────────────────┤
│ OVERALL METRICS:                                        │
│   Total Predictions:    10,000                          │
│   Correct Predictions:   9,050                          │
│   Overall Accuracy:      90.5%  ✅                      │
├─────────────────────────────────────────────────────────┤
│ SLICE-BASED METRICS:                                    │
│                                                         │
│   Group A (Majority):                                   │
│     Population:          9,000                          │
│     Correct:             8,550                          │
│     Accuracy:            95.0%  ✅                      │
│                                                         │
│   Group B (Minority):                                   │
│     Population:          1,000                          │
│     Correct:               500                          │
│     Accuracy:            50.0%  ❌ CRITICAL ALERT       │
├─────────────────────────────────────────────────────────┤
│ FAIRNESS METRICS:                                       │
│   Disparate Impact Ratio: 0.55 ❌ (Threshold: 0.80)     │
│   Performance Gap:        45 percentage points          │
└─────────────────────────────────────────────────────────┘
```

### Calculation Breakdown

```
Overall Accuracy Calculation:
----------------------------
Group A Correct = 9,000 × 0.95 = 8,550
Group B Correct = 1,000 × 0.50 =   500
                                ------
Total Correct   =                9,050

Overall Accuracy = 9,050 / 10,000 = 90.5%
```

### Key Insight

```
⚠️ HIDDEN PROBLEM REVEALED:
   
   What stakeholders see:  "90.5% accuracy - Excellent!"
   What's actually happening: Group B gets coin-flip predictions
   
   Without slice monitoring: Problem stays hidden
   With slice monitoring:    Problem immediately visible
```

---

## 🔢 Example 2: Healthcare Diagnostic AI

### Input Data

```
Total Patients: 10,000
Urban Patients: 8,000
Rural Patients: 2,000
```

### Performance Results

```
┌─────────────────────────────────────────────────────────┐
│           HEALTHCARE DIAGNOSTIC AI METRICS              │
├─────────────────────────────────────────────────────────┤
│ OVERALL METRICS:                                        │
│   Sensitivity (TPR):     88.8%  ✅                      │
│   Specificity (TNR):     91.2%  ✅                      │
│   Missed Diagnoses:      11.2%                          │
├─────────────────────────────────────────────────────────┤
│ BY LOCATION:                                            │
│                                                         │
│   Urban Patients (n=8,000):                             │
│     Sensitivity:         94.0%  ✅                      │
│     Specificity:         96.0%  ✅                      │
│     Missed Diagnoses:     6.0%                          │
│                                                         │
│   Rural Patients (n=2,000):                             │
│     Sensitivity:         68.0%  ❌ WARNING              │
│     Specificity:         72.0%  ❌ WARNING              │
│     Missed Diagnoses:    32.0%  ❌ CRITICAL             │
└─────────────────────────────────────────────────────────┘
```

### Key Insight

```
⚠️ UNEQUAL HARM:

   Rural patients have 5x higher missed diagnosis rate!
   
   Urban: 6% missed → 480 patients
   Rural: 32% missed → 640 patients
   
   Despite being 20% of population, rural patients 
   account for 57% of missed diagnoses.
```

---

## 🔢 Example 3: Simpson's Paradox - Resume Screening

### Performance Over Time

```
┌─────────────────────────────────────────────────────────┐
│        RESUME SCREENING AI - MONTHLY TREND              │
├─────────────────────────────────────────────────────────┤
│ Month │ Overall │ Female Slice │ Male Slice │ Status   │
├───────┼─────────┼──────────────┼────────────┼──────────┤
│   1   │  89.0%  │    86.0%     │   90.0%    │   OK     │
│   2   │  90.0%  │    84.0%     │   92.0%    │   OK     │
│   3   │  91.0%  │    80.0%     │   94.0%    │ ⚠️WARN   │
│   4   │  91.0%  │    75.0%     │   96.0%    │ ⚠️WARN   │
│   5   │  90.0%  │    70.0%     │   97.0%    │ ❌ALERT  │
└─────────────────────────────────────────────────────────┘

SIMPSON'S PARADOX IN ACTION:
- Overall accuracy: STABLE (89% → 90%)
- Female slice:     DEGRADING (86% → 70%)
- Male slice:       IMPROVING (90% → 97%)

Without slice monitoring: "Performance is stable!"
With slice monitoring:    "Female candidates being harmed!"
```

---

## 🔢 Example 4: Fairness Metrics Calculation

### Disparate Impact Ratio

```
Scenario: Loan Approval Rates
- Group A approval rate: 80%
- Group B approval rate: 60%

Disparate Impact Ratio = P(Approved | Group B) / P(Approved | Group A)
                       = 0.60 / 0.80
                       = 0.75

Result: 0.75 < 0.80 threshold → ❌ DISPARATE IMPACT DETECTED
```

### Equal Opportunity Difference

```
Scenario: True Positive Rates
- Group A TPR: 92%
- Group B TPR: 68%

Equal Opportunity Difference = |TPR_A - TPR_B|
                             = |0.92 - 0.68|
                             = 0.24

Result: 0.24 > 0.10 threshold → ❌ UNEQUAL OPPORTUNITY DETECTED
```

---

## 📊 Dashboard Visualization

```
╔═══════════════════════════════════════════════════════════╗
║          SLICE-BASED MONITORING DASHBOARD                 ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  📈 OVERALL PERFORMANCE                                   ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ Accuracy: ████████████████████░░░░ 90.5%            │  ║
║  │ Status: ✅ Above threshold (85%)                    │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                           ║
║  📊 SLICE BREAKDOWN                                       ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ Group A: ███████████████████░ 95.0% ✅              │  ║
║  │ Group B: ██████████░░░░░░░░░░ 50.0% ❌ ALERT        │  ║
║  │ Age 18-30: █████████████████░ 92.0% ✅              │  ║
║  │ Age 31-50: ████████████████░░ 88.0% ✅              │  ║
║  │ Age 51+: █████████████░░░░░░░ 68.0% ⚠️              │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                           ║
║  ⚖️ FAIRNESS METRICS                                      ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ Disparate Impact:    0.55 ❌ (Min: 0.80)            │  ║
║  │ Equal Opportunity:   0.24 ❌ (Max: 0.10)            │  ║
║  │ Demographic Parity:  0.18 ⚠️ (Max: 0.10)            │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                           ║
║  🚨 ACTIVE ALERTS: 3                                      ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ [CRITICAL] Group B accuracy below 60% threshold     │  ║
║  │ [HIGH] Disparate Impact Ratio below 0.80            │  ║
║  │ [MEDIUM] Age 51+ trending downward (-3% this week)  │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 🎯 Summary of Outputs

| Scenario | Overall Metric | Hidden Problem | Detection Method |
|----------|----------------|----------------|------------------|
| Loan Model | 90.5% accuracy | 50% for minorities | Slice-based accuracy |
| Healthcare | 88.8% sensitivity | 32% missed for rural | Location-based slicing |
| Resume AI | Stable overall | Female slice degrading | Temporal + demographic slicing |
| Fairness | N/A | DIR = 0.55 | Fairness metric calculation |

---

> **Key Takeaway**: Every example shows that aggregate metrics hide critical disparities. Slice-based monitoring reveals the truth.
