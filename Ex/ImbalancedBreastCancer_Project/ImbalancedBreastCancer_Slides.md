
# Imbalanced Breast Cancer Analysis
## Research Briefing & Implementation Review

---

# 1. Project Overview & Steps
**Objective**: Build a robust SVM/DT classifier for a 90:10 Imbalanced Dataset.

**Steps Followed**:
1.  **Data Engineering**: 
    *   Loaded Breast Cancer Data (Benign vs Malignant).
    *   Undersampled Malignant class to force a strict 90:10 imbalance ratio.
2.  **Model Configuration**:
    *   Implemented SVM (Linear) and Decision Tree.
    *   Configured three variants for each: Baseline, Balanced Weights, Custom Weights {0:9, 1:1}.
3.  **Validation**:
    *   Stratified Split (80/20).
    *   Metrics: Recall (Sensitivity), Precision, F1-Score, ROC-AUC.

---

# 2. Key Concept: The Imbalance Problem
**What it is**: 
When 'Healthy' patients outnumber 'Sick' patients by 9 to 1.

**Why it matters**: 
Standard algorithms optimize for *Average Accuracy*.
*   If we predict "Everyone is Healthy", we get 90% Accuracy.
*   But we miss 100% of the Cancer.

**Solution**: 
*   **Class Weights**: Penalize the model 9x more for missing a cancer case than for a false alarm.
*   **Metric Shift**: Move from Accuracy to **Recall** (Sensitivity).

---

# 3. Execution Output & Results

| Model Variation | Precision | Recall (Key) | F1-Score | Note |
| :--- | :--- | :--- | :--- | :--- |
| **SVM Baseline** | High | **Low (~0.0)** | Low | Fails to detect cancer. |
| **SVM Balanced** | Moderate | **High (>0.8)** | High | **Success**. |
| **DT Baseline** | High | Low/Mid | Moderate | Unstable. |
| **DT Custom** | Moderate | High | High | Good alternative. |

**Observation**: 
The "Baseline" models fell into the accuracy trap. They ignored the minority class because it was "cheaper" to just predict the majority.

---

# 4. Detailed Observations
*   **Recall is the Vital Sign**: 
    *   We observed that Weighted SVMs consistently flagged the malignant cases.
    *   Trade-off: This increased False Positives (Healthy people flagged as sick). In oncology, this is acceptable; missing a tumor is not.
    
*   **Hyperplane Behavior**:
    *   The SVM Baseline pushed the boundary deep into the malignant territory to satisfy the Benign majority.
    *   The **Weighted SVM** pushed the boundary back, creating a wider safety margin around the cancer cases.

---

# 5. Conclusion & Recommendation
**Final Verdict**: 
Deploy **SVM with Custom/Balanced Weights**.

**Why?**
1.  **Safety**: It maximizes Recall, ensuring patient safety.
2.  **Robustness**: SVM margins generalize better than Decision Trees on small, imbalanced datasets.
3.  **Adjustability**: The weight parameter allows doctors to tune the sensitivity preference without retraining.
