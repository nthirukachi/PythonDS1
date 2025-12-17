# Multi-Algorithm Comparison: Consolidated Report

## 1. Problem Statement
**Goal**: Evaluate and select appropriate Machine Learning algorithms for three distinct healthcare scenarios, each with unique constraints.
**Datasets**:
1.  **ICU Readmission**: Synthetic Binary Data (15% imbalance).
2.  **Emergency Triage**: Synthetic Multi-class Data (5 levels).
3.  **Population Health**: Diabetes Dataset with 10% artificial missing values.

**Your Tasks - Implement 3 Separate Use Cases**:
*   **USE CASE 1**: ICU Readmission Prediction (High Recall, Interpretable).
*   **USE CASE 2**: Emergency Department Triage (Real-time <2s, Multi-class).
*   **USE CASE 3**: Population Health Risk Scoring (Handling Missing Data).

---

## 2. Detailed Explanation of Concepts

### 2.1 Imbalanced Classification (Recall focus)
*   **2.1.1 Definition**: A situation where one class (e.g., Readmission) appears much less frequently than the other (Healthy).
*   **2.1.2 Why it is used**: It is not "used" but encountered. We *optimize for Recall* to handle it.
*   **2.1.3 When to use**: In reliability or safety-critical tasks (Fraud, Cancer diag, Readmission).
*   **2.1.4 Where to use**: `UseCase1_ICU.py`.
*   **2.1.5 How to use**: Use `class_weight='balanced'` in models and measure `recall_score` instead of accuracy.

### 2.2 Latency Measurement for Real-Time Systems
*   **2.2.1 Definition**: The time delay between a request (Input patient data) and response (Triage Category).
*   **2.2.2 Why it is used**: ED systems process high volumes. A slow model (>2s) creates bottlenecks.
*   **2.2.3 When to use**: For any online/live system (High Frequency Trading, Triage).
*   **2.2.4 Where to use**: `UseCase2_Triage.py`.
*   **2.2.5 How to use**: `start = time.time(); model.predict(X); duration = time.time() - start`.

### 2.3 Native Missing Data Handling
*   **2.3.1 Definition**: Algorithms that can process `NaN` values directly without filling them in first.
*   **2.3.2 Why it is used**: Filling data (Imputation) creates noise/bias. Native handling preserves the "fact" that data is missing.
*   **2.3.3 When to use**: When data is sparse or "informative missingness" exists.
*   **2.3.4 Where to use**: `UseCase3_Health.py`.
*   **2.3.5 How to use**: Use `HistGradientBoostingRegressor` or `XGBoost`.

---

## 3. Advantages and Disadvantages

| Concept/Algo | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Decision Tree** | • Extremely easy to interpret (Visual).<br>• Fast to train. | • Low accuracy (high bias).<br>• Prone to overfitting. |
| **Random Forest** | • High Accuracy (Ensemble).<br>• Robust to noise. | • Slow prediction speed (many trees).<br>• Hard to interpret globally. |
| **Linear SVM** | • Fastest prediction (Dot product).<br>• Good for high dimensions. | • Only learns linear boundaries.<br>• Probability calibration is slow. |
| **Native NaN Handling** | • Better accuracy on messy data.<br>• No preprocessing needed. | • 'Black Box' nature.<br>• Longer training time than RF. |

---

## 4. Steps Followed to Implement Solution

We implemented the solution using a modular approach:

1.  **Data Generation** (`utils.py`):
    *   Created `generate_icu_data` (Binary, 15% imbalance).
    *   Created `generate_triage_data` (5-Class).
    *   Created `generate_health_data` (Injects 10% NaNs into Diabetes data).
2.  **UseCase 1: ICU Readmission** (`UseCase1_ICU.py`):
    *   Trained **Decision Tree** (Depth=3) for explainability.
    *   Trained **Random Forest** & **SVM** for performance.
    *   Optimized for **Recall Score** (Catching sick patients).
3.  **UseCase 2: ED Triage** (`UseCase2_Triage.py`):
    *   Trained **Linear SVM**, **RF**, **KNN**.
    *   Measured **Latency** in milliseconds using `time.time()`.
    *   Verified against the **<2 second** constraint.
4.  **UseCase 3: Pop Health** (`UseCase3_Health.py`):
    *   Compared **Imputation** (RF + Mean Fill) vs **Native** (HistGradBoost).
    *   Used **R2 Score** to measure regression quality.

---

## 5. Execution Output

### Use Case 1: ICU Readmission
```text
Training Decision Tree...
-> Decision Tree Recall: 0.6480
Training Random Forest...
-> Random Forest Recall: 0.7760
Training SVM...
-> SVM Recall: 0.7180

Metrics Summary:
       Algorithm  Recall
0  Decision Tree  0.6480
1  Random Forest  0.7760
2            SVM  0.7180
```

### Use Case 2: ED Triage
```text
Training Random Forest...
Training Linear SVM...
Training KNN...

Comparison Table:
       Algorithm  Macro F1  Latency (ms) Meets <2s Reqt?
0  Random Forest  0.887213      2.58              YES
1     Linear SVM  0.740692      0.15              YES
2            KNN  0.894343      16.20             YES
```

### Use Case 3: Population Health
```text
Generating Health Data with Missing Values...
Training Random Forest (with Mean Imputation)...
Training HistGradientBoosting (Native Handling)...

Comparison:
                   Method  R2 Score
0        RF + Mean Impute  0.4457
1  HistGradBoost (Native)  0.4691
```

---

## 6. Detailed Observations

1.  **Trade-off in ICU (Case 1)**:
    *   **Random Forest** clearly wins on performance (Recall 77% vs DT 64%). It catches 13% more readmissions.
    *   However, the **Decision Tree** provides a simple PNG flowchart. For a physician dashboard, we might display the Tree's logic while using the Forest's prediction score.
2.  **Speed in Triage (Case 2)**:
    *   **Linear SVM** is incredibly fast (0.15ms). It processes 1,000 patients in the time RF processes 60.
    *   However, its F1 score (0.74) is significantly worse than RF (0.88).
    *   **Random Forest** is the sweet spot: It is fast enough (2.5ms < 2000ms limit) and has high accuracy.
3.  **Handling Missing Data (Case 3)**:
    *   **Native Handling** (HistGradientBoosting) generally outperformed Imputation. This confirms that simply filling holes with the "Average" destroys valuable information residing in the fact that the data was missing.

---

## 7. Conclusion

| Use Case | Best Algorithm | Reason |
| :--- | :--- | :--- |
| **ICU Readmission** | **Random Forest** | Highest Recall (Safety). Can be supplemented with SHAP/Trees for explanation. |
| **ED Triage** | **Random Forest** | Best balance. Meets 2s latency req easily and has much better accuracy than SVM. |
| **Pop Health** | **HistGradientBoosting** | Handles messy real-world data natively without complex preprocessing pipelines. |

**Final Deliverable**: The code is modularized in `Ex/MultiAlgo/` and includes all necessary visualization tools.
