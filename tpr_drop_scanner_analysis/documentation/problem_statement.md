# Problem Statement: TPR Drop After New Scanner Introduction

## 🧩 The Problem

You observe that **slice-wise TPR (True Positive Rate) drops sharply for one site** after a new scanner is introduced, while **service latency and error rate remain normal**.

### Scenario Details
- **Before**: Site 3 had TPR of ~0.92 (similar to other sites)
- **After**: New scanner introduced at Site 3
- **Observation**: Site 3 TPR drops to ~0.65
- **Service Metrics**: All healthy (latency ~50ms, error rate <0.1%)
- **Other Sites**: TPR unchanged (~0.90+)

---

## 🎯 Why This Matters

### In Real-World Medical Imaging
- **False Negatives are dangerous**: Missing disease in screening
- **Silent failures**: System appears healthy but predictions are wrong
- **Unequal harm**: Patients at Site 3 receiving worse care
- **Regulatory risk**: Compliance issues if not detected

### In Production ML Systems
- **Overall accuracy hides subgroup issues**
- **New data sources can break models silently**
- **Slice-based monitoring is essential**

---

## 🔍 Key Questions to Answer

### 1. Why is this a Data/Model Issue (Not Service Issue)?
- What indicators prove infrastructure is healthy?
- What indicators prove model quality has degraded?
- Why does the new scanner cause this specific problem?

### 2. What Diagnostics Should We Run?
- Minimum 4 diagnostic approaches
- How to identify root cause
- How to quantify the distribution shift

### 3. What Mitigations Should We Prioritize?
- Minimum 3 mitigation strategies
- At least one safe-fallback option
- Short-term vs long-term solutions

---

## 🪜 Steps to Solve the Problem

### Step 1: Understand the Data Flow
1. Multiple sites with different scanners send images
2. ML model processes images and returns predictions
3. Predictions are used for downstream decisions

### Step 2: Identify the Anomaly
1. Monitor overall accuracy → appears normal
2. Monitor slice-wise TPR → reveals Site 3 degradation
3. Compare with service metrics → confirms infrastructure is healthy

### Step 3: Diagnose the Root Cause
1. Compare feature distributions (KL Divergence)
2. Analyze prediction confidence (calibration check)
3. Break down confusion matrices per scanner
4. Track temporal TPR trends vs scanner rollout

### Step 4: Apply Mitigations
1. **Immediate**: Safe fallback (human review routing)
2. **Short-term**: Preprocessing normalization
3. **Long-term**: Domain adaptation / model retraining

---

## 🎯 Expected Output

After completing this analysis, you will:

1. **Understand** why service health ≠ model health
2. **Detect** covariate shift using multiple diagnostics
3. **Implement** safe fallback mechanisms
4. **Improve** model robustness through domain adaptation
5. **Monitor** slice-based metrics to catch future issues

---

## 📊 Success Criteria

| Metric | Before Analysis | After Mitigation |
|--------|-----------------|------------------|
| Site 3 TPR | ~0.65 (degraded) | ~0.90+ (recovered) |
| Detection Time | Unknown | Immediate via slice alerts |
| Patient Safety | At risk | Protected via human review |
| Model Robustness | Fragile | Domain-adapted |

---

## 🔗 Related Concepts
- Covariate Shift
- Distribution Drift
- Slice-based Monitoring
- Safe Fallback / Human-in-the-Loop
- Domain Adaptation
- Model Calibration
