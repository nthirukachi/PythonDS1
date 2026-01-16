# Concepts Explained

## 1. Data Drift vs Concept Drift

### 🔹 Data Drift ($P(X)$)
- **Definition**: The statistical properties of the input data ($X$) change. The distribution of variables shifts.
- **Example**: In our loan model, if salaries suddenly double due to inflation, the input distribution shifts right.
- **Formula**: $P(X_{train}) \neq P(X_{test})$
- **Detection**: Statistical tests like **KS Test** (Kolmogorov-Smirnov) or **PSI** (Population Stability Index).
- **Does it break the model?**: Maybe. The model might extrapolate poorly to data it hasn't seen before.

### 🔹 Concept Drift ($P(Y|X)$)
- **Definition**: The relationship between inputs ($X$) and the target ($Y$) changes. The "Ground Truth" logic changes.
- **Example**: The bank changes its policy. An income of \$50k used to be approved ($Y=1$), now it is rejected ($Y=0$).
- **Formula**: $P(Y|X_{train}) \neq P(Y|X_{test})$
- **Detection**: Needs labeled data (Ground Truth). Monitor **Accuracy**, **F1-Score**, or **Error Rate**.
- **Does it break the model?**: YES. The old model is now effectively predicting wrong based on old rules.

---

## 2. Statistical Checks

### 🔹 Kolmogorov-Smirnov (KS) Test
- **What**: A non-parametric test that compares two probability distributions.
- **How**: It measures the maximum distance ($D$) between the cumulative distribution functions (CDF) of the two samples.
- **Output**:
    - **Statistic**: The distance $D$.
    - **P-Value**: Probability that the two samples come from the same distribution.
    - **Rule**: If P-Value < 0.05, we reject the null hypothesis (assume distributions are DIFFERENT).

### 🔹 Data Quality Checks
- **Null Check**: Ensuring no missing data enters the model (Model will crash or impute badly).
- **Range Check**: Ensuring values are physically possible (e.g., Age cannot be negative, Income cannot be -500).

---

## 3. Alerting Strategy ("Human in the Loop")
ML Systems cannot fix themselves effectively in all cases.
- **If Data Drift**: We might need to retrain OR we might just need to check if the data pipeline is broken.
- **If Concept Drift**: We DEFINITELY need to retrain with new labels.

**Our Alert Rule**:
> "IF KS-Test P-Value < 0.05 OR Mean Shift > 20%, THEN Trigger Retraining Alert."
