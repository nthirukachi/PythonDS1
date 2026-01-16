# Concepts Explained: TPR Drop Analysis

## Table of Contents
1. [TPR (True Positive Rate) / Recall](#1-tpr-true-positive-rate--recall)
2. [Slice-based Monitoring](#2-slice-based-monitoring)
3. [Covariate Shift](#3-covariate-shift)
4. [Service Metrics vs Model Metrics](#4-service-metrics-vs-model-metrics)
5. [KL Divergence](#5-kl-divergence)
6. [Prediction Confidence Calibration](#6-prediction-confidence-calibration)
7. [Safe Fallback / Human-in-the-Loop](#7-safe-fallback--human-in-the-loop)
8. [Domain Adaptation](#8-domain-adaptation)
9. [Preprocessing Normalization](#9-preprocessing-normalization)

---

## 1. TPR (True Positive Rate) / Recall

### Definition
**TPR (True Positive Rate)**, also called **Recall** or **Sensitivity**, measures how well a model identifies positive cases.

```
TPR = TP / (TP + FN)

Where:
- TP = True Positives (correctly predicted positives)
- FN = False Negatives (missed positives)
```

### Why It Matters
- **Medical imaging**: TPR tells us what percentage of disease cases we correctly detect
- **High TPR is critical**: Missing disease (False Negative) can be life-threatening
- **FNR = 1 - TPR**: False Negative Rate is the complement

### Real-Life Analogy
> **Office Example**: If your security system should detect 100 intruders, and it only catches 85, your TPR is 85%. The 15 missed intruders are **False Negatives**.

### When to Use
- When **missing positives is costly** (medical diagnosis, fraud detection)
- When you care about **sensitivity** over precision
- When false negatives have **higher impact** than false positives

### How to Calculate
```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tpr = tp / (tp + fn)
```

---

## 2. Slice-based Monitoring

### Definition
**Slice-based monitoring** tracks model performance across different **data subgroups** (slices), rather than just looking at overall accuracy.

### Why It Matters
- **Overall accuracy can hide subgroup harm**
- **Unequal performance** across demographics, devices, or regions
- **Fairness and compliance** requirements

### Real-Life Analogy
> **Student Exam Example**: If a class average is 85%, but students from one school score only 60%, the overall average **hides the problem** for that subgroup.

### When to Use
- Production ML systems with **diverse data sources**
- Applications with **fairness requirements**
- Systems where **different subgroups** may behave differently

### How to Implement
```python
# Evaluate per slice
for scanner in df['scanner'].unique():
    slice_data = df[df['scanner'] == scanner]
    # Calculate TPR for this slice
    tpr = calculate_tpr(slice_data)
    print(f"{scanner}: TPR = {tpr}")
```

---

## 3. Covariate Shift

### Definition
**Covariate shift** occurs when the **input data distribution changes** between training and production, while the relationship between inputs and outputs remains the same.

### Why It Matters
- **New scanner** produces images with different characteristics
- **Model trained on old data** fails on new data
- **Silent failure**: Model is confident but wrong

### Visual Summary
```
Training Data:          Production Data (New Scanner):
┌──────────────┐        ┌──────────────────┐
│  Mean: 100   │   →    │    Mean: 120     │
│  Std: 15     │        │    Std: 25       │
│  Clean       │        │    Noisy         │
└──────────────┘        └──────────────────┘
              ↓
         MODEL FAILS ON NEW DISTRIBUTION!
```

### Real-Life Analogy
> **Teaching Example**: If you train students to answer exam questions in English, but the actual exam is in French, they will fail even if the content is the same.

### When to Use This Knowledge
- Deploying models to **new regions, devices, or data sources**
- **Continuous monitoring** for distribution drift
- **Retraining decisions** based on detected shift

---

## 4. Service Metrics vs Model Metrics

### Definition

| Service Metrics | Model Metrics |
|-----------------|---------------|
| Latency, Error Rate, Uptime | Accuracy, TPR, Precision |
| Infrastructure health | Prediction quality |
| "Is the system running?" | "Are predictions correct?" |

### Why It Matters
- **Normal service metrics ≠ Healthy model**
- Service can be **fast and reliable** while making **wrong predictions**
- This is a **silent failure mode**

### Key Insight
```
┌───────────────────────────────────────────────────────────────┐
│  Service Latency: 50ms ✅     →  Infrastructure is HEALTHY   │
│  Error Rate: 0.1% ✅          →  No crashes or timeouts      │
│  TPR (Site 3): 0.65 ❌        →  Model quality DEGRADED      │
│                                                               │
│  CONCLUSION: Data/Model issue, NOT service issue             │
└───────────────────────────────────────────────────────────────┘
```

### Real-Life Analogy
> **Office Report Example**: The printer is working perfectly fast (service healthy), but the reports it prints contain incorrect data (model unhealthy).

---

## 5. KL Divergence

### Definition
**Kullback-Leibler (KL) Divergence** measures how different one probability distribution is from another reference distribution.

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

Where:
- P = New distribution (new scanner)
- Q = Reference distribution (training data)
```

### Why It Matters
- **Quantifies distribution shift** numerically
- **Higher KL = More different** distributions
- Can **automate drift detection** with thresholds

### How to Interpret
| KL Divergence | Interpretation |
|---------------|----------------|
| < 0.05 | Distributions are similar |
| 0.05 - 0.1 | Minor shift, monitor |
| > 0.1 | Significant shift, investigate |

### How to Calculate
```python
from scipy.stats import entropy

# Create histograms
hist_original, bins = np.histogram(original_data, bins=30, density=True)
hist_new, _ = np.histogram(new_data, bins=bins, density=True)

# Add epsilon to avoid division by zero
hist_original += 1e-10
hist_new += 1e-10

# Normalize
hist_original /= hist_original.sum()
hist_new /= hist_new.sum()

# Calculate KL Divergence
kl_div = entropy(hist_new, hist_original)
```

### Advantages
- ✅ Mathematically rigorous
- ✅ Can be computed per-feature
- ✅ Enables automated alerting

### Disadvantages
- ❌ Sensitive to binning choices
- ❌ Not symmetric (KL(P||Q) ≠ KL(Q||P))
- ❌ Can be infinite if distributions don't overlap

---

## 6. Prediction Confidence Calibration

### Definition
**Calibration** measures whether a model's confidence scores match actual accuracy. A well-calibrated model with 80% confidence should be correct 80% of the time.

### Why It Matters
- **Miscalibrated model** is confident but wrong
- **New scanner** may cause over-confident wrong predictions
- **Safe fallback** relies on confidence thresholds

### Key Diagnostic
```
Scanner_A (original):
  - Average Confidence on False Negatives: 0.25 (low, expected)

Scanner_C (new):
  - Average Confidence on False Negatives: 0.65 (HIGH!)
  - ⚠️ Model is CONFIDENTLY WRONG on new scanner!
```

### Real-Life Analogy
> **Exam Confidence Example**: A student says "I'm 95% sure my answer is correct" but is actually wrong 40% of the time. They are **overconfident and miscalibrated**.

---

## 7. Safe Fallback / Human-in-the-Loop

### Definition
**Safe fallback** routes uncertain or low-confidence predictions to human experts for review, preventing automated harm.

### Why It's Critical
- **Immediate protection** while investigating root cause
- **No harm from wrong automated decisions**
- **Regulatory compliance** in healthcare, finance

### How to Implement
```python
def safe_fallback(prediction, confidence, threshold=0.7):
    if confidence < threshold:
        return "ROUTE_TO_HUMAN_REVIEW"
    else:
        return prediction
```

### Priority: P0 (Highest)
This is the **first mitigation to implement** because:
1. Can be deployed **immediately**
2. **Zero risk** of harming patients
3. Buys time for proper fix

### Real-Life Analogy
> **Hospital Example**: When automated diagnostic is uncertain, the case goes to a senior doctor for final decision, ensuring patient safety.

---

## 8. Domain Adaptation

### Definition
**Domain adaptation** techniques help a model generalize from the training domain (source) to a new domain (target) with different data characteristics.

### Approaches
1. **Fine-tuning**: Retrain model with some target domain data
2. **Transfer learning**: Adapt pre-trained features
3. **Domain-adversarial training**: Learn domain-invariant features

### Why It Works
- Model learns **features specific to new scanner**
- Reduces **distribution gap** between training and production
- **Long-term fix** for covariate shift

### How to Implement (Fine-tuning)
```python
# Collect labeled samples from new scanner
new_scanner_samples = collect_labeled_samples(new_scanner, n=200)

# Combine with original training data
combined_training = original_data + new_scanner_samples

# Retrain model
model.fit(combined_training)
```

### Advantages
- ✅ Addresses root cause
- ✅ Improves model robustness
- ✅ Reduces future drift issues

### Disadvantages
- ❌ Requires labeled data from new domain
- ❌ Takes time to collect and validate
- ❌ May require model retraining pipeline

---

## 9. Preprocessing Normalization

### Definition
**Preprocessing normalization** applies scanner-specific transformations to align new scanner data with the training distribution.

### Techniques
1. **Histogram matching**: Match pixel intensity distributions
2. **Z-score normalization**: Standardize to reference mean/std
3. **Contrast normalization**: Adjust brightness/contrast

### How to Implement
```python
# Reference statistics from training data
ref_mean = original_scanner_data.mean()
ref_std = original_scanner_data.std()

# Normalize new scanner data
new_mean = new_scanner_data.mean()
new_std = new_scanner_data.std()

normalized = (new_scanner_data - new_mean) / new_std * ref_std + ref_mean
```

### Advantages
- ✅ Quick to implement
- ✅ No model retraining needed
- ✅ Can be applied immediately

### Disadvantages
- ❌ May not capture all differences
- ❌ Assumes linear relationship
- ❌ Not a permanent fix

---

## 💼 Interview Perspective

### Common Questions

1. **Q: What's the difference between service issues and model issues?**
   - A: Service issues affect infrastructure (latency, errors). Model issues affect prediction quality (accuracy, TPR). Normal service metrics don't guarantee healthy model predictions.

2. **Q: How do you detect distribution shift?**
   - A: Use KL divergence, feature histograms, or embedding analysis to compare production data against training distribution.

3. **Q: Why implement slice-based monitoring?**
   - A: Overall accuracy hides subgroup issues. A model can have 95% overall accuracy but 60% TPR for a specific demographic.

4. **Q: What's your first action when TPR drops for one site?**
   - A: Implement safe fallback (route to human review) immediately, then investigate root cause.

5. **Q: When should you retrain vs preprocess?**
   - A: Preprocess for quick fixes. Retrain for long-term robustness if you have labeled data from the new domain.
