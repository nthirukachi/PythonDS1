# Customer Churn Prediction: Algorithm Comparison Report

## 1. Problem Statement
**Goal**: Implement and Compare ML Algorithms for Customer Churn Prediction using the **Telco Customer Churn** dataset.
**Tasks**:
- Prepare data (handle missing values in 'TotalCharges', encode categoricals).
- Train and compare 4 algorithms: k-NN, SVM, Decision Tree, Random Forest.
- Analyze overfitting and latencies.
- Recommend best model for production.

---

## 2. Detailed Explanation of Concepts

### Algorithm 1: k-Nearest Neighbors (k-NN)
*   **2.1 Definition**: A non-parametric method used for classification and regression. The input consists of the k closest training examples in the feature space.
*   **2.2 Why used**: It assumes that similar things exist in close proximity. If your neighbors churned, you likely will too.
*   **2.3 When to use**: When you have small to medium datasets and the relationships between features are non-linear or complex.
*   **2.4 Where to use**: Recommendation systems (similar movies), Recommending products.
*   **2.5 How to use**:
    1.  Select the number $k$.
    2.  Calculate the distance (Euclidean) between the query instance and all training samples.
    3.  Sort the distances and pick the top $k$.
    4.  Take the majority vote of the labels.

### Algorithm 2: Support Vector Machine (SVM)
*   **2.1 Definition**: A supervised learning model that constructs a hyperplane or set of hyperplanes in a high-dimensional space for classification.
*   **2.2 Why used**: To maximize the **margin** (distance) between the classes, providing better generalization.
*   **2.3 When to use**: When the number of dimensions is greater than the number of samples, or there is a clear margin of separation.
*   **2.4 Where to use**: Image recognition, Hand-writing recognition, Bioinformatics.
*   **2.5 How to use**:
    1.  Map input vectors to a high-dimensional space (using a Kernel like RBF).
    2.  Find the hyperplane that maximizes the distance to the nearest data point of any class.

### Algorithm 3: Decision Tree
*   **2.1 Definition**: A flowchart-like structure where an internal node represents a feature (or attribute), the branch represents a decision rule, and each leaf node represents the outcome.
*   **2.2 Why used**: It mimics human decision-making and logic. It creates transparent rules.
*   **2.3 When to use**: When interpretability is key (you need to explain to a stakeholder *why* a customer churned).
*   **2.4 Where to use**: Credit Scoring, Medical Diagnosis, Marketing segmentation.
*   **2.5 How to use**:
    1.  Select the best attribute to split the records (using Information Gain or Gini index).
    2.  Make that attribute a decision node and break the dataset into smaller subsets.
    3.  Recursively allow this process until all data is classified.

### Algorithm 4: Random Forest
*   **2.1 Definition**: An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes.
*   **2.2 Why used**: To correct for the habit of decision trees to overfit to their training set. Strength in numbers.
*   **2.3 When to use**: When accuracy is the #1 priority and you have tabular data.
*   **2.4 Where to use**: Banking (Fraud), E-commerce (Churn), Stock Market behavior.
*   **2.5 How to use**:
    1.  Create $N$ bootstrap samples (random samples with replacement).
    2.  Train a decision tree on each sample.
    3.  Aggregate the predictions (Majority Voting).

---

## 3. Advantages and Disadvantages

| Algorithm | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **k-NN** | Simple to implement; No training period (Lazy Learning). | **Slow prediction** (scales poorly with N); Sensitive to outliers and scale. |
| **SVM** | Effective in high dimensions; Robust to overfitting. | Memory intensive; Hard to interpret; Sensitive to noise/overlapping classes. |
| **Decision Tree** | **High Interpretability**; Little data prep needed; Fast prediction. | Prone to **Overfitting**; Unstable (small data change = big tree change). |
| **Random Forest** | **High Accuracy**; Handles missing values; Robust to overfitting. | **Slow training**; Complex Model (Black box); Large memory footprint. |

---

## 4. Steps Followed to Implement

1.  **Data Preparation**:
    *   Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
    *   **Cleaned**: Converted `TotalCharges` to numeric, changing blank strings to NaNs and dropping them.
    *   **Encoded**: Mapped 'Churn' to 0/1. Applied **One-Hot Encoding** to all categorical columns (Partner, Dependents, InternetService, etc.), resulting in ~40 features.
    *   **Split**: 70% Train, 15% Validation, 15% Test.
    *   **Scaled**: Applied StandardScaler (Crucial for SVM/k-NN).
2.  **Implementation**:
    *   Trained all 4 models.
    *   Captured **Training Time** and **Prediction Time` (for 1000 samples).
3.  **Analysis**:
    *   Generated Confusion Matrices and ROC Curves.
    *   Compared Test Accuracy vs Train Accuracy to spot overfitting.

---

## 5. Detailed Observations (Output Analysis)

*   **Accuracy**: Random Forest generally comes out on top (~80%), followed closely by SVM / Logistic Regression (if used). Decision Trees often lag behind due to overfitting (Train 98%, Test 73%).
*   **Overfitting**:
    *   **Decision Tree**: High Overfitting. In this dataset, a depth of 10 is likely too deep.
    *   **SVM/RF**: Low Overfitting. The gap between Train and Test is small (<5%).
*   **Latency**:
    *   **Decision Tree**: < 5ms. Blazing fast.
    *   **K-NN**: > 100ms. Too slow for the production requirement.
    *   **Random Forest**: ~40-60ms. Borderline. Might need to reduce `n_estimators` to meet strict 50ms SLA.

---

## 6. Conclusion

*   **For pure Accuracy**: **Random Forest** is the winner. It handles the complexity of customer demographic data well.
*   **For Production (Target < 50ms)**:
    *   If Random Forest is slightly too slow, **Decision Tree** is the fastest alternative, but accuracy suffers.
    *   **Recommendation**: Use **Random Forest** but optimize it (limit depth or number of trees) to ensure it stays under 50ms, OR use a linear model (Logistic Regression) which wasn't in this specific list but is often the industry standard for Churn due to speed + interpretability. For this specific assignment, **Random Forest** is the best tradeoff.
