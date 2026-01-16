# 📚 Concepts Explained

## 1. k-Nearest Neighbors (k-NN)
### Definition
A simple, instance-based supervised learning algorithm used for classification and regression. It classifies a new data point based on the majority class of its 'k' closest neighbors.

### Why is it used?
- **Simplicity:** Easy to understand and implement.
- **Non-parametric:** Makes no assumptions about the underlying data distribution.

### When to use it?
- Small to medium-sized datasets.
- When the decision boundary is irregular.

### How it works
1.  **Store:** Keep all training data.
2.  **Distance:** Calculate the distance (e.g., Euclidean) between the new point and all training points.
3.  **Sort:** Find the `k` nearest points.
4.  **Vote:** The majority class among these `k` neighbors becomes the prediction.

---

## 2. Bias-Variance Tradeoff
### Definition
The balance between two sources of error that affect algorithm performance:
- **Bias:** Error due to overly simplistic assumptions (Underfitting).
- **Variance:** Error due to excessive sensitivity to small fluctuations in the training set (Overfitting).

### In the context of k-NN:
- **Low k (e.g., k=1):** **High Variance**, Low Bias. The model captures every noisy detail. Complex boundary.
- **High k (e.g., k=100):** **Low Variance**, High Bias. The model smooths out everything. Simple boundary.

---

## 3. Class Imbalance
### Definition
A situation where the classes are not represented equally.
- **Majority Class:** 95% (Safe Transactions)
- **Minority Class:** 5% (Fraud)

### Why is it a problem?
Standard algorithms aim to maximize Accuracy. In a 95:5 split, a model that simply predicts "Safe" for *everything* achieves 95% accuracy but fails its purpose (finding Fraud).

### Solutions used here:
1.  **Stratification:** Ensuring the train/test split keeps the same 19:1 ratio.
2.  **Recall Metric:** Focusing on how many Frauds we caught, rather than overall accuracy.
3.  **Threshold Tuning:** changing the decision rule (e.g., flag fraud if > 20% neighbors are fraud, not 50%).

---

## 4. Libraries Explaination

### `make_classification` (sklearn.datasets)
- **What:** Generates a random n-class classification problem.
- **Why:** To create controlled synthetic data satisfying specific properties (like 19:1 imbalance) for testing.

### `train_test_split` (sklearn.model_selection)
- **What:** Splits data into random train and test subsets.
- **Stratify argument:** Critical for imbalance. It ensures the 5% fraud ratio is preserved in *both* training and testing sets, so we don't accidentally end up with a test set having zero fraud.

### `KNeighborsClassifier` (sklearn.neighbors)
- **What:** The implementation of the k-NN vote.
- **Arguments:**
    - `n_neighbors`: The 'k'.
    - `weights`: 'uniform' (default) or 'distance' (closer neighbors have more say).
    - `predict_proba`: Returns probability estimates (e.g., 0.2 Safe, 0.8 Fraud) instead of just the class.

### `accuracy_score` vs `recall_score`
- **Accuracy:** (TP + TN) / Total. Good for balanced classes.
- **Recall (Sensitivity):** TP / (TP + FN). The ratio of *actual* Frauds that were correctly identified. **Critical for Fraud Detection.**
