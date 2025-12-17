"""
Part3_AlgoLevel.py
-------------------------------------------------------------------------------
Part 3: Algorithm-Level Solutions

PROBLEM STATEMENT:
Modifying data (SMOTE) is "fake". In medicine, we often prefer to keep real data.
Instead, we can modify the ALGORITHM:
1.  **Class Weights**: Penalize the model 5x more for missing a sick person.
2.  **Threshold Tuning**: Lower the bar for classification.
    Standard: Predict Sick if Prob > 0.50.
    Tuned: Predict Sick if Prob > 0.30.

STEPS TO SOLVE:
1. Train RF with `class_weight='balanced'`.
2. Compare custom weights vs balanced weights.
3. Use `precision_recall_curve` to scan all possible thresholds.
4. Select a threshold that guarantees 80% Recall.

EXPECTED OUTPUT:
- Comparison Table.
- Precision-Recall Curve (PNG).
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, PrecisionRecallDisplay
from utils import generate_imbalanced_data

def run_part3_algolevel():
    print("\n=== Part 3: Algorithm-Level Solutions ===")
    
    X_train, X_test, y_train, y_test = generate_imbalanced_data()
    results = []
    
    # ---------------------------------------------------------
    # 1. Class Weights
    # ---------------------------------------------------------
    # WHAT: List of weight strategies to test.
    # 'balanced': Auto-calculates weights inversely proportional to frequency.
    # {0:1, 1:5}: Manually says "Class 1 is 5x more important".
    weights_to_test = [
        ('Balanced', 'balanced'),
        ('Custom 1:5', {0: 1, 1: 5})
    ]
    
    for name, w in weights_to_test:
        print(f"Training with Weight: {name}...")
        
        # WHAT: Initialize model with weight parameter.
        clf = RandomForestClassifier(n_estimators=50, class_weight=w, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        results.append({
            'Method': f"RF ({name})",
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        })
        
    print(pd.DataFrame(results))

    # ---------------------------------------------------------
    # 2. Threshold Tuning (The most powerful tool)
    # ---------------------------------------------------------
    print("\nPerforming Threshold Tuning...")
    
    # WHAT: Train a weighted model first.
    clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # WHAT: Get Probabilities instead of Labels.
    # predict_proba returns [Prob_0, Prob_1]. We want Prob_1.
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # WHAT: Calculate P/R Curve.
    # Computes Precision/Recall for every unique probability in y_probs.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # WHAT: Find the threshold that yields Recall >= 0.80.
    target_recall = 0.80
    
    # Logic:
    # 1. 'recalls' array goes from 1.0 down to 0.0.
    # 2. We look for the last index where recall >= 0.80.
    # 3. Use np.where returns indices meeting condition.
    idx = np.where(recalls >= target_recall)[0][-1] 
    
    optimal_thresh = thresholds[idx]
    op_prec = precisions[idx]
    op_rec = recalls[idx]
    
    print(f"Goal: Ensure we catch {target_recall*100}% of sick patients.")
    print(f"Optimal Threshold Found: {optimal_thresh:.4f} (Standard is 0.50)")
    print(f"Resulting Precision: {op_prec:.4f}")
    
    # WHAT: Plot the Curve.
    plt.figure(figsize=(8, 6))
    disp = PrecisionRecallDisplay(precision=precisions, recall=recalls)
    disp.plot()
    
    plt.title(f"PR Curve (Optimal Threshold = {optimal_thresh:.2f})")
    
    # Plot markers
    plt.axhline(y=op_prec, color='r', linestyle='--', label=f'Precision {op_prec:.2f}')
    plt.axvline(x=op_rec, color='g', linestyle='--', label=f'Recall {op_rec:.2f}')
    plt.legend()
    
    plt.savefig('PR_Curve_Threshold.png')
    print("Saved PR_Curve_Threshold.png")
    plt.close()
    
    return optimal_thresh

if __name__ == "__main__":
    run_part3_algolevel()
