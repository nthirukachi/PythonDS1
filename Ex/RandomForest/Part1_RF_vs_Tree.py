"""
Part1_RF_vs_Tree.py
-------------------------------------------------------------------------------
Part 1: Build Random Forest vs Decision Tree

PROBLEM STATEMENT:
A single Decision Tree is simple but fragile. It tends to "Overfit" (memorize noise).
A **Random Forest** is an "Ensemble" of many trees. It fixes the fragility of a 
single tree by averaging multiple opinions (Bagging).
We want to empirically PROVE that the Forest is better than the Tree using metrics.

STEPS TO SOLVE:
1. Load prepared data (from utils).
2. Train a Single Decision Tree (constrained to depth 10).
3. Train a Random Forest (100 trees, each depth 10).
4. Predict on the Test Set.
5. Calculate Metrics: Accuracy, Precision, Recall, F1.

EXPECTED OUTPUT:
- Comparison Table showing RF > DT in F1 Score.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_loan_data

def get_metrics(y_true, y_pred, model_name):
    """
    Helper function to calculate common classification metrics.
    
    ARGS:
    - y_true: The actual answers (Ground Truth).
    - y_pred: The model's guesses.
    - model_name: String label for the table.
    """
    # WHAT: Calculate Accuracy (Correct / Total).
    acc = accuracy_score(y_true, y_pred)
    
    # WHAT: Calculate Precision (True Positives / All Predicted Positives).
    # WHY: "If model says RISK, how often is it right?"
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    # WHAT: Calculate Recall (True Positives / All Actual Positives).
    # WHY: "Did we catch all the risky people?"
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # WHAT: Calculate F1 (Harmonic Mean of Precision and Recall).
    # WHY: Best single metric for imbalanced data.
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

def run_part1_comparison():
    print("\n=== Part 1: Single Tree vs Random Forest ===")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, _ = load_loan_data()
    
    results = []
    
    # ---------------------------------------------------------
    # 2. Single Decision Tree
    # ---------------------------------------------------------
    print("Training Single Decision Tree (max_depth=10)...")
    
    # WHAT: Initialize Decision Tree Classifier.
    # ARGUMENTS:
    # - max_depth=10: Limit tree height to prevent total overfitting.
    # - random_state=42: Deterministic behavior.
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # WHAT: Train the model.
    dt.fit(X_train, y_train)
    
    # WHAT: Get metrics dictionary using our helper.
    metrics_dt = get_metrics(y_test, dt.predict(X_test), 'Decision Tree')
    results.append(metrics_dt)
    
    # ---------------------------------------------------------
    # 3. Random Forest
    # ---------------------------------------------------------
    print("Training Random Forest (100 Trees)...")
    
    # WHAT: Initialize Random Forest Classifier.
    # ARGUMENTS:
    # - n_estimators=100: Create 100 independent decision trees.
    # - max_depth=10: Each tree is limited to depth 10 (fair comparison).
    # - n_jobs=-1: Use ALL available CPU cores (Parallel processing).
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # WHAT: Train the ensemble (fits 100 trees).
    rf.fit(X_train, y_train)
    
    metrics_rf = get_metrics(y_test, rf.predict(X_test), 'Random Forest')
    results.append(metrics_rf)
    
    # 4. Compare
    df_res = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df_res)
    
    print("\nEXPLANATION:")
    print("Random Forest outperforms because it reduces VARIANCE.")
    print("Interpretation: A single tree might make a mistake on edge cases.")
    print("100 trees vote, and the majority vote is usually correct.")
    
    # Return objects for use in Part 2
    return rf, X_train, X_test, y_test

if __name__ == "__main__":
    run_part1_comparison()
