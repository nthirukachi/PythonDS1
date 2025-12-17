"""
Part1_Baseline.py
-------------------------------------------------------------------------------
Part 1: Demonstrate Class Imbalance Problem

PROBLEM STATEMENT:
If we train a standard Random Forest on imbalanced data, it will likely fail
to detect the minority class (Sick Patients).
However, it will report "High Accuracy" because it correctly identifies the
vast majority of healthy people.
We need to demonstrate this dangerous paradox.

STEPS TO SOLVE:
1. Load 85/15 imbalanced data.
2. Train a Random Forest with default settings (weights=None).
3. Evaluate using Confusion Matrix and Classification Report.
4. Highlight the low Recall for Class 1.

EXPECTED OUTPUT:
- High Accuracy (>85%).
- Low Recall for Class 1 (<20%).
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import generate_imbalanced_data

def run_part1_baseline():
    print("\n=== Part 1: Baseline (Ignoring Imbalance) ===")
    
    # 1. Get Data
    X_train, X_test, y_train, y_test = generate_imbalanced_data()
    
    # ---------------------------------------------------------
    # Train Standard Model
    # ---------------------------------------------------------
    print("Training Standard Random Forest...")
    
    # WHAT: Initialize Standard RF.
    # ARGUMENTS:
    # - n_estimators=100: Standard forest size.
    # - class_weight=None: (Default) Treat every error equally.
    #   THIS IS THE PROBLEM. Missing a Class 1 (Sick) is treated same as missing Class 0.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Use standard fit.
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    
    print("\nConfusion Matrix:")
    # WHAT: Print the raw counts [TN, FP, FN, TP].
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    # WHAT: Print metrics per class.
    # Look at row '1' (The Sick Class). Check column 'recall'.
    print(classification_report(y_test, y_pred))
    
    print("OBSERVATION:")
    print("Notice Class 1 (Sick) Recall is likely very low.")
    print("The model biases towards Class 0 (Healthy) to maximize global accuracy.")
    
    return model, X_train, X_test, y_test

if __name__ == "__main__":
    run_part1_baseline()
