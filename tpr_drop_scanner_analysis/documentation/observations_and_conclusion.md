# Observations and Conclusion

## 📊 Execution Results

### Slice-Wise TPR Evaluation

| Scanner | Site | TPR | FPR | Status |
|---------|------|-----|-----|--------|
| Scanner_A | Site_1 | 0.91 | 0.08 | ✅ Normal |
| Scanner_B | Site_2 | 0.90 | 0.09 | ✅ Normal |
| Scanner_C_NEW | Site_3 | 0.65 | 0.12 | 🔴 **DEGRADED** |

### Service Health Check

| Metric | Value | Status |
|--------|-------|--------|
| Average Latency | ~50ms | ✅ Healthy |
| P99 Latency | ~65ms | ✅ Healthy |
| Error Rate | 0.1% | ✅ Healthy |
| Infrastructure | Operational | ✅ Healthy |

---

## 🔍 Key Observations

### Observation 1: TPR Drop is Isolated
- **Only Site 3** (new scanner) shows TPR degradation
- Other sites maintain TPR > 0.90
- **Conclusion**: Not a global service failure

### Observation 2: Service Metrics Are Normal
- Latency and error rate unchanged
- Requests are being processed successfully
- **Conclusion**: Infrastructure is healthy

### Observation 3: Distribution Shift Detected
- KL Divergence for feature_0: **0.35** (significant shift)
- New scanner mean: 120 vs Original mean: 100
- New scanner std: 25 vs Original std: 15
- **Conclusion**: Covariate shift from new scanner

### Observation 4: Model is Confidently Wrong
- Average confidence on False Negatives: **0.65** for new scanner
- Model is making wrong predictions with high confidence
- **Conclusion**: Miscalibration on new distribution

### Observation 5: Temporal Correlation Confirmed
- TPR was stable at ~0.92 before scanner introduction
- TPR dropped to ~0.65 exactly when new scanner was deployed
- **Conclusion**: Causal relationship established

---

## 💡 Insights for Production ML

### Insight 1: Overall Accuracy Hides Subgroup Issues
If we only monitored overall accuracy, we would NOT have detected this issue. The degradation at Site 3 was masked by normal performance at Sites 1 and 2.

### Insight 2: Normal Service ≠ Healthy Model
This is a **silent failure mode**. The system appears completely healthy from an infrastructure perspective, but the model is causing harm to a specific subgroup.

### Insight 3: New Data Sources Require Validation
Introducing a new scanner, device, or data source should trigger:
- Distribution comparison against training data
- Validation testing before full deployment
- Close monitoring during rollout

### Insight 4: Safe Fallback is Critical
Having a human-in-the-loop fallback mechanism allows you to:
- Immediately protect affected users
- Buy time for proper investigation
- Maintain service while fixing the issue

---

## ✅ Mitigation Effectiveness

### Safe Fallback (Human Review Routing)
- **Result**: 23% of new scanner predictions routed to human review
- **Impact**: Protected patients from incorrect automated decisions
- **Recommendation**: Keep active until model is retrained

### Preprocessing Normalization
- **Result**: Feature means aligned (120 → 100)
- **Impact**: Partial improvement in TPR
- **Recommendation**: Use as temporary measure

### Domain Adaptation (Fine-tuning)
- **Result**: TPR on new scanner improved from 0.65 to 0.88
- **Impact**: Significant recovery of model performance
- **Recommendation**: Deploy after validation

---

## 🎯 Conclusion

### Root Cause
The TPR drop is caused by **covariate shift** from the new scanner producing images with different characteristics than the training data. This is a **data/model health issue**, not a service issue.

### Evidence Summary
1. ✅ Service metrics NORMAL → Infrastructure healthy
2. 🔴 TPR drop ISOLATED to Site 3 → Not global failure
3. 📊 KL Divergence HIGH → Distribution shift confirmed
4. ⚠️ High confidence on FN → Model miscalibrated
5. 📅 Temporal correlation → New scanner is cause

### Recommended Action Plan
| Priority | Action | Timeline |
|----------|--------|----------|
| P0 | Deploy safe fallback (human review) | Immediate |
| P1 | Apply preprocessing normalization | 1-2 days |
| P2 | Collect labeled data from new scanner | 1 week |
| P3 | Retrain model with domain adaptation | 2 weeks |
| P4 | Implement automated slice-level alerting | Ongoing |

### Key Takeaway
> **Always monitor slice-wise metrics, not just overall accuracy. Service health does not guarantee model health. Implement safe fallbacks for high-stakes applications.**

---

## 📚 References
- Covariate Shift in Machine Learning
- Slice-based Fairness Monitoring
- Human-in-the-Loop AI Systems
- Domain Adaptation Techniques
