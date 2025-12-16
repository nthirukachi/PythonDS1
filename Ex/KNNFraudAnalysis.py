"""
Problem Statement:
Fraud Detection System Analysis (k-NN).
Context:
- Class Imbalance: 19:1 (95% Legitimate, 5% Fraud).
- Cost: False Negative (Missing Fraud) is very expensive.
- Observed Data: Accuracy peaks at k=9.

Sub-Problems:
1. Part A: Select k (Recommend k=9 with caveats) and provide 3 technical reasons.
2. Part B: Explain Bias-Variance Tradeoff relative to k.
3. Part C: Propose 2 modifications for the 19:1 Imbalance.

Steps to Solve:
1. Simulation: Generate imbalanced data (950 Safe, 50 Fraud).
2. Experiment: Run k-NN for k=1 to 20.
3. Validation: Observe Accuracy peaking (likely around 9-10) vs Recall.
4. Print Answers: Detailed text blocks answering the specific prompt questions.

Expected Output:
- Performance matrix showing Accuracy vs Recall for different k.
- Text answers explaining the selection of k=9 for stability, the Bias-Variance curve, and Imbalance solutions.
"""

"""
Task: Part A (Selecting k)
----------------------------------------
Question: Which k would you choose and provide three technical reasons? consider cost.
Answer: I would choose k=9 (The sweet spot).
Reason 1 (Variance Reduction): Low k (e.g., k=1) captures too much noise (outliers). In fraud, a single weird legitimate transaction shouldn't flag the card. k=9 smooths this noise.
Reason 2 (Generalization): The peak accuracy at k=9 indicates the model has minimized the Total Error (Bias^2 + Variance + Irreducible Error).
Reason 3 (Business Context): While low k (e.g., 3) might have slightly higher Recall, it often floods the system with False Positives. k=9 provides a stable baseline, which we can then tune for Recall using the modifications in Part C.
"""

"""
Task: Part B (Bias-Variance Tradeoff)
----------------------------------------
Question: Explain the bias-variance tradeoff. Why does accuracy peak at k=9?
Answer: 
- k=1 to 8 (High Variance): The model is too complex. It overfits to local data points (noise). Error is high because it's unstable.
- k=9 (Optimum): The "Goldilocks" zone. We have averaged out enough noise (Variance reduced) without ignoring the underlying pattern (Bias is still low).
- k > 9 (High Bias): The model becomes too simple. It starts voting for the majority class (Safe) everywhere, ignoring local Fraud pockets. Accuracy drops (or plateaus at 95% baseline) as meaningful structure is washed out.
"""

"""
Task: Part C (Imbalance Modifications)
----------------------------------------
Question: Propose two specific modifications for 19:1 imbalance.
Modification 1: Distance-Weighted k-NN.
- How: Give closer neighbors more voting power (Weight = 1/distance). 
- Why: Even if only 2 of 9 neighbors are Fraud, if they are *very close* (tight cluster), their weighted vote can outweigh the 7 distant Safe neighbors. This preserves local minority clusters.

Modification 2: Adjusted Decision Threshold.
- How: Instead of requiring >50% vote (5/9), flag fraud if >20% (2/9) neighbors are Fraud.
- Why: Directly addresses the "Cost" of missing fraud. It trades some False Positives for much higher Recall, ensuring we catch the expensive fraud cases.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, accuracy_score

# ==========================================
# 1. Simulation (Generating the Bias-Variance Curve)
# ==========================================

print("--- 1. Data Simulation (19:1 Imbalance) ---")

# Generate Data
# What: 2000 samples, 5% (0.05) fraud.
# Output: X features and y labels.
X, y = make_classification(
    n_samples=2000, 
    n_features=10, 
    n_informative=5, 
    weights=[0.95, 0.05], # 19:1 Imbalance
    random_state=42
)

# Split
# Why: Standard validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train Imbalance: {np.bincount(y_train)} (Safe/Fraud)")

# ==========================================
# 2. Analyzing k (1 to 20)
# ==========================================

print("\n--- 2. k-NN Performance Analysis ---")

results = []

# Loop through k
# Why: To find the "Peak" mentioned in the prompt.
for k in range(1, 21, 2): # Odd numbers only to avoid ties
    # k-NN Standard
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Metrics
    # Accuracy: Overall correctness.
    # Recall: Ability to find Fraud (Class 1).
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({'k': k, 'Accuracy': acc, 'Recall': rec})

# Convert to DataFrame for nice printing
# Output: Table of k vs Scores.
df_res = pd.DataFrame(results)
print(df_res.set_index('k'))

# ==========================================
# 3. Demonstrating Part C (Modifications)
# ==========================================

print("\n--- 3. Part C Demonstration (weighted vs threshold) ---")

# Modification 1: Distance Weighted
# What: weights='distance'
# Why: Closer points vote harder.
knn_weighted = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn_weighted.fit(X_train, y_train)
y_pred_w = knn_weighted.predict(X_test)
rec_w = recall_score(y_test, y_pred_w)
print(f"Mod 1: Weighted k=9 Recall: {rec_w:.4f} (Improved Local Sensitivity)")

# Modification 2: Threshold Tuning
# What: Check probability manually.
knn_prob = KNeighborsClassifier(n_neighbors=9) # Standard weights
knn_prob.fit(X_train, y_train)
probs = knn_prob.predict_proba(X_test)[:, 1] # Prob of Fraud

# Custom Threshold > 0.2 (2 out of 9 votes)
y_pred_cust = (probs >= 0.2).astype(int)
rec_cust = recall_score(y_test, y_pred_cust)
print(f"Mod 2: Threshold > 0.2 Recall: {rec_cust:.4f} (Maximizing Safety)")
