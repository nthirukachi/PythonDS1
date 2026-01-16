# 📚 Concepts Explained

## 1. Principal Component Analysis (PCA)

### Definition
A dimensionality reduction technique that transforms features into a smaller set of uncorrelated "principal components" that capture most of the variance.

### Why is it used?
- **Curse of Dimensionality:** In high dimensions (50 features), Euclidean distance becomes meaningless. Points appear equally far apart. k-NN relies on distance, so it fails.
- **Noise Removal:** Many of 50 features might be irrelevant. PCA keeps only the most informative combinations.

### When to use it?
- When you have many features relative to samples.
- When k-NN, SVM, or clustering performance degrades.

### How it works (Simple Analogy)
Imagine 50 exam subjects. PCA finds that "Overall Science Aptitude" (a mix of Physics, Chemistry, Biology) explains most of the grade variation. Instead of 50 scores, you now track 15 "aptitude scores."

---

## 2. SMOTE (Synthetic Minority Over-sampling Technique)

### Definition
An oversampling algorithm that creates *new* synthetic examples of the minority class by interpolating between existing minority samples.

### Why is it used?
- **Class Imbalance:** 950 vs 50 means k-NN will almost always vote "Healthy" (majority class).
- **Generates Diversity:** Unlike simple duplication, SMOTE creates *new* points, reducing overfitting.

### How it works
1.  Pick a minority sample.
2.  Find its k-nearest minority neighbors.
3.  Draw a line between the sample and a neighbor.
4.  Create a new point randomly along that line.

### When to use it?
- When the minority class is severely underrepresented.
- Applied **only to the training set** (never the test set).

---

## 3. Data Leakage

### Definition
When information from outside the training dataset (e.g., test data or future data) is used to create the model, leading to overly optimistic performance estimates.

### Why is it dangerous?
The model appears to perform well during testing but fails in real-world deployment.

### How we prevent it here
We split the data into Train/Test **before** applying SMOTE. If we applied SMOTE first, synthetic minority samples in the training set could be derived (interpolated) from points that end up in the test set.

---

## 4. Choosing k for k-NN

### Rule of Thumb
A common heuristic is `k = sqrt(N)` or a small percentage of N (e.g., 1-5%).

### In this problem
- N ≈ 1000 training samples.
- 5% of 1000 = 50.
- We choose **k = 51** (odd number to avoid ties in voting).

### Why odd?
With an even `k`, you might have a 50-50 tie (e.g., 25 Healthy, 25 Disease). Odd `k` forces a winner.

---

## 5. Evaluation Metrics for Imbalanced Data

### Accuracy (Misleading Here)
`(TP + TN) / Total`. A model predicting "Healthy" for everyone achieves 95% accuracy but is useless.

### Recall (Sensitivity)
`TP / (TP + FN)`. How many actual Disease cases did we catch? **Critical for medical diagnosis.**

### Precision
`TP / (TP + FP)`. Of all predicted Disease, how many were correct?

### F1-Score
`2 * (Precision * Recall) / (Precision + Recall)`. Harmonic mean balancing both.

### Confusion Matrix
A 2x2 table showing TP, TN, FP, FN. Essential for understanding where the model fails.
