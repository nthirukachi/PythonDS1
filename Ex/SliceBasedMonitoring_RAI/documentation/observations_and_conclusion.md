# 📝 Observations and Conclusion

---

## 🔍 Key Observations

### Observation 1: Overall Metrics Create False Security

| Finding | Evidence |
|---------|----------|
| High accuracy ≠ fair model | 90% overall can hide 50% for minorities |
| Majority groups dominate | 90% majority at 95% + 10% minority at 50% = 90.5% |

---

### Observation 2: Slice Selection Is Critical

- Choose slices based on domain knowledge
- Include protected groups
- Ensure sufficient data per slice

---

### Observation 3: Fairness Metrics Are Complementary

| Metric | Focus |
|--------|-------|
| Disparate Impact | Outcome ratios |
| Equal Opportunity | True Positive Rates |
| Equalized Odds | Both TPR and FPR |

---

### Observation 4: Training Bias Propagates

Historical bias → Model learns → Reproduces bias → Reinforces bias

---

## 📊 Example: Loan Approval

| Metric | Group A | Group B | Overall |
|--------|---------|---------|---------|
| Population | 9,000 | 1,000 | 10,000 |
| Accuracy | 95% | 50% | 90.5% |
| Approval Rate | 82% | 45% | 78.3% |

**Key Finding:** Disparate Impact = 45/82 = 0.55 (below 0.8 threshold)

---

## 🎯 Key Takeaways

1. **Always compute slice-level metrics**
2. **Set per-slice thresholds**
3. **Track trends over time**
4. **Implement automated alerting**

---

## 📌 Conclusion

> **Overall accuracy is necessary but not sufficient.** Without slice-based monitoring, hidden harm to specific groups goes undetected.

| Without Monitoring | With Monitoring |
|-------------------|-----------------|
| Hidden disparities | Visible disparities |
| Compliance risk | Demonstrable fairness |

---

## 💼 Interview Quick Reference

| Question | Answer |
|----------|--------|
| What is slice-based monitoring? | Evaluating performance separately per subgroup |
| Why overall accuracy fails? | Larger groups dominate the average |
| Key fairness metric? | Disparate Impact Ratio ≥ 0.8 |
