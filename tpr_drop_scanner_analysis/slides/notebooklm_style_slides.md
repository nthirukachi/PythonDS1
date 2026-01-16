# NotebookLM-Style Slides: TPR Drop After New Scanner Introduction

---

## Slide 1: Title & Objective

# TPR Drop Analysis After New Scanner Introduction

### 🎯 Objective
Understand why **slice-wise TPR drops sharply** for one site after a new scanner is introduced, while service metrics remain normal.

### Learning Goals
- Distinguish **data/model issues** from **service issues**
- Master **4 diagnostic approaches**
- Implement **3 mitigation strategies**

---

## Slide 2: Problem Statement

# The Problem

> Slice-wise TPR drops sharply for one site after a new scanner is introduced

### Key Facts
| Before | After | Service Metrics |
|--------|-------|-----------------|
| Site 3 TPR: 0.92 | Site 3 TPR: 0.65 | Latency: Normal ✅ |
| All sites similar | Only Site 3 drops | Error Rate: Normal ✅ |

### The Question
**Is this a service issue or a data/model issue?**

---

## Slide 3: Real-World Use Case

# Medical Imaging Scenario

### Context
- **3 hospital sites** with different scanners
- **ML model** for disease detection
- **New scanner** deployed at Site 3

### The Risk
- **False Negatives = Missed disease**
- **Patients at Site 3** receiving worse care
- **System appears healthy** but predictions are wrong

### Industry Relevance
- Healthcare, Radiology, Pathology
- Any ML system with diverse data sources

---

## Slide 4: Input Data / Inputs

# Data Overview

### Multi-Site Scanner Data
```
📊 1000 samples per scanner

Scanner_A (Site_1): Original, Mean=100, Std=15
Scanner_B (Site_2): Similar to A, Mean=100, Std=15
Scanner_C (Site_3): NEW, Mean=120, Std=25 + Noise
```

### Feature Structure
- 10 numeric features (simulated image characteristics)
- Binary labels (disease present/absent)
- Scanner and site metadata

### Key Difference
**New scanner has DIFFERENT distribution!**

---

## Slide 5: Concepts Used (High Level)

# Key Concepts

| Concept | Purpose |
|---------|---------|
| **TPR (Recall)** | Measure positive class detection |
| **Covariate Shift** | Input distribution change |
| **Slice-based Monitoring** | Per-subgroup evaluation |
| **KL Divergence** | Quantify distribution difference |
| **Safe Fallback** | Human-in-the-loop protection |
| **Domain Adaptation** | Model robustness improvement |

---

## Slide 6: Concepts Breakdown (Simple)

# Simple Explanations

### TPR = True Positive Rate
> "Out of 100 disease cases, how many did we correctly detect?"

### Covariate Shift
> "Training data looks different from production data"

### Slice-based Monitoring
> "Check accuracy for each group separately, not just overall"

### Safe Fallback
> "When unsure, let a human expert decide"

---

## Slide 7: Step-by-Step Solution Flow

# Solution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. OBSERVE                                                 │
│     ↓ TPR drops for Site 3                                  │
│     ↓ Service metrics normal                                │
│                                                             │
│  2. DIAGNOSE                                                │
│     ↓ Feature distribution comparison                       │
│     ↓ Confidence calibration check                          │
│     ↓ Confusion matrix breakdown                            │
│     ↓ Temporal trend analysis                               │
│                                                             │
│  3. MITIGATE                                                │
│     ↓ P0: Safe fallback (human review)                      │
│     ↓ P1: Preprocessing normalization                       │
│     ↓ P2: Domain adaptation / retraining                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Slide 8: Code Logic Summary

# Code Structure

### Main Sections
1. **Simulate Data**: 3 scanners with different distributions
2. **Train Model**: Only on original scanners (A, B)
3. **Evaluate per Slice**: TPR for each scanner
4. **Check Service Metrics**: Confirm infrastructure healthy
5. **Run Diagnostics**: 4 diagnostic functions
6. **Apply Mitigations**: 3 mitigation strategies

### Key Functions
- `simulate_scanner_data()`
- `train_model_on_original_scanners()`
- `evaluate_per_scanner()`
- `diagnostic_*()` functions
- `mitigation_*()` functions

---

## Slide 9: Important Functions & Parameters

# Key Functions

### `evaluate_per_scanner(df, model, scaler)`
- Calculates TPR per scanner slice
- Returns confusion matrix breakdown

### `diagnostic_feature_distribution(df)`
- Computes KL divergence per feature
- Threshold > 0.1 = significant shift

### `mitigation_safe_fallback(df, model, scaler, confidence_threshold=0.7)`
- Routes low-confidence predictions to human review
- Default threshold: 0.7

---

## Slide 10: Execution Output

# Results

### TPR per Scanner
| Scanner | TPR | Status |
|---------|-----|--------|
| Scanner_A | 0.91 | ✅ Normal |
| Scanner_B | 0.90 | ✅ Normal |
| Scanner_C_NEW | 0.65 | 🔴 DEGRADED |

### Service Metrics
| Metric | Value | Status |
|--------|-------|--------|
| Latency | 50ms | ✅ Healthy |
| Error Rate | 0.1% | ✅ Healthy |

### Diagnostic: KL Divergence
- feature_0: **0.35** (high shift!)

---

## Slide 11: Observations & Insights

# Key Insights

### 1. Overall Accuracy Hides Subgroup Issues
> If we only monitored overall accuracy, we would NOT have detected this issue

### 2. Normal Service ≠ Healthy Model
> System appears healthy but predictions are wrong (silent failure)

### 3. New Scanner = Distribution Shift
> KL Divergence confirms the new scanner data is very different

### 4. Model is Confidently Wrong
> High confidence on false negatives = miscalibration

---

## Slide 12: Advantages & Limitations

# Analysis Summary

### Advantages of This Approach
| ✅ Advantage |
|-------------|
| Detects silent failures |
| Quantifies distribution shift |
| Provides actionable mitigations |
| Enables safe deployment |

### Limitations
| ❌ Limitation |
|--------------|
| Requires labeled data for adaptation |
| KL divergence sensitive to binning |
| Human review doesn't scale infinitely |
| Preprocessing may not capture all differences |

---

## Slide 13: Interview Key Takeaways

# Interview Prep

### Q1: Why is this a data/model issue, not service issue?
> "Service metrics (latency, error rate) are normal, indicating healthy infrastructure. The TPR drop is isolated to one site/scanner, pointing to data distribution shift."

### Q2: What diagnostics would you run?
> "Feature distribution comparison (KL divergence), confidence calibration check, per-scanner confusion matrices, and temporal trend analysis."

### Q3: What's your first mitigation?
> "Safe fallback - route low-confidence predictions to human review for immediate protection."

### Q4: Why implement slice-based monitoring?
> "Overall accuracy hides subgroup issues. A model can have 95% accuracy but 60% TPR for a specific demographic."

---

## Slide 14: Conclusion

# Summary

### Root Cause
**Covariate shift** from new scanner with different data characteristics

### Evidence
- ✅ Service metrics normal
- 🔴 TPR drop isolated to Site 3
- 📊 High KL divergence confirmed
- 📅 Temporal correlation with scanner rollout

### Action Plan
| Priority | Action |
|----------|--------|
| P0 | Safe fallback (human review) |
| P1 | Preprocessing normalization |
| P2 | Domain adaptation / retraining |

### Key Takeaway
> **Always monitor slice-wise metrics. Service health ≠ model health. Implement safe fallbacks for high-stakes applications.**

---

*End of Presentation*
