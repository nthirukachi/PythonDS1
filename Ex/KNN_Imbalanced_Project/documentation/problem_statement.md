# 🧩 Problem Statement

## 1. The Core Problem
**Goal:** Detect a **rare disease** from patient data using the k-Nearest Neighbors classifier.

**Context:**
- **Dataset:** 1,000 patients, each with 50 medical features (e.g., blood markers, age, BMI).
- **Imbalance:** Only 50 patients (5%) have the disease. 950 (95%) are healthy.
- **Challenge:** Standard k-NN will ignore the minority class and predict "Healthy" for everyone.

## 2. Why is this Important?
- **Medical Diagnosis:** Missing a disease (False Negative) can be life-threatening.
- **High Dimensions:** With 50 features, distance metrics become unreliable ("Curse of Dimensionality").
- **Real-World:** This pattern (rare event detection) applies to fraud, network intrusion, and manufacturing defects.

## 3. 🪜 Steps to Solve the Problem
1.  **Generate Data:** Create a synthetic 19:1 imbalanced dataset.
2.  **Split Data:** Train/Test split *before* any preprocessing (avoid leakage).
3.  **Reduce Dimensions:** Apply PCA (50 → 15 features).
4.  **Balance Classes:** Apply SMOTE to the training set only.
5.  **Train k-NN:** Use k=51 (5% of N, odd number to break ties).
6.  **Evaluate:** Focus on Recall and F1-Score for Class 1 (Disease).

## 4. 🎯 Expected Output
- A **Confusion Matrix** showing True Positives, False Negatives, etc.
- A **Classification Report** with Precision, Recall, and F1-Score.
- Significant improvement in detecting the rare disease class.
