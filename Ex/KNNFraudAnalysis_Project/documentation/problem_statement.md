# 🧩 Problem Statement

## 1. The Core Problem
**Goal:** Detect fraudulent credit card transactions using the **k-Nearest Neighbors (k-NN)** algorithm.

**Context:**
- **Imbalanced Data:** Real-world fraud is rare. Our dataset has a **19:1 ratio** (95% Legitimate, 5% Fraud).
- **Cost Sensitivity:** Missing a fraud case (False Negative) is extremely expensive for the bank (lost money, reputation). False Positives (customer annoyance) are less costly but still strictly monitored.
- **Model Tuning:** We need to find the optimal `k` (number of neighbors) that balances stability (Variance) and ability to capture patterns (Bias).

## 2. Why is this Important?
- **Financial Loss:** Banks lose billions annually to fraud.
- **Trust:** Customers leave banks that allow fraud on their accounts.
- **Complexity:** Simple accuracy is misleading. A model that says "All transactions are Safe" has 95% accuracy but is **100% useless** for fraud detection.

## 3. 🪜 Steps to Solve the Problem
1.  **Simulation:** Generate a synthetic dataset mimicking the 19:1 imbalance.
2.  **Experimentation:** Train k-NN models with `k` ranging from 1 to 20.
3.  **Analysis:** Observe how Accuracy and Recall (fraud detection rate) change with `k`.
4.  **Optimization:**
    - Identify the optimal `k`.
    - Propose and implement modifications (Distance Weighting, Threshold Tuning) to improve Fraud detection.

## 4. 🎯 Expected Output
1.  A performance table showing Accuracy and Recall for various `k` values.
2.  Technical justification for choosing `k=9`.
3.  Explanation of the Bias-Variance tradeoff in this specific context.
4.  Demonstration of how **Distance Weighting** and **Threshold Adjustment** significantly improve Recall.
