"""
Part5_Imbalance.py
-------------------------------------------------------------------------------
Part 5: Address Class Imbalance with Advanced Techniques

PROBLEM STATEMENT:
Standard models struggle with 0.17% fraud because they favor the majority class.
We must use specialized techniques to "force" the model to find frauds.

STEPS TO SOLVE:
We implement and compare 4 distinct strategies:
1. SMOTE (Oversampling): Generate fake fraud data.
2. Class Weights (Cost-sensitive): Penalize missing frauds.
3. Threshold Tuning (Post-processing): Lower the alert bar.
4. Balanced Random Forest (Ensemble): Algorithm-level fix.

CONCEPTS & ARGUMENTS:
1. SMOTE(sampling_strategy=0.1): Increases fraud count until it is 10% of legit count.
2. Threshold Tuning: Moving the cutoff from 0.5 to e.g. 0.2 to catch more fraud.
   - Tradeoff: Increases Recall (Catch more) but lowers Precision (More false alarms).

EXPECTED OUTPUT:
- Comparison table showing how each technique impacts Recall and Precision.
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
# WHAT: Libraries for Imbalance.
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from Part2_DataPrep import prepare_data

def evaluate_technique(model, X_test, y_test, name, threshold=0.5):
    """
    Evaluates model with a specific threshold.
    """
    # 1. Get Probabilities (Confidence of fraud 0.0 to 1.0)
    # model.predict_proba returns [[Prob_0, Prob_1], ...]
    # We select column 1 (Fraud Probability).
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 2. Apply Threshold Logic
    # WHAT: If probability >= threshold, mark as 1 (Fraud).
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'Technique': name,
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'Threshold Used': threshold
    }

def run_imbalance_handling():
    print("\n=== Part 5: Address Class Imbalance ===")
    
    data = prepare_data()
    if data is None: return
    X_train, _, X_test, y_train, _, y_test, _ = data
    
    results = []
    
    # ---------------------------------------------------------
    # Technique 1: SMOTE (Synthetic Minority Oversampling)
    # ---------------------------------------------------------
    print("[Technique 1] SMOTE...")
    # WHAT: Create synthetic neighbors.
    # ARGUMENT: sampling_strategy=0.1
    # WHY: We don't want 50/50 balance (too much fake data).
    #      We aim for 1:10 ratio (Fraud is 10% of Legit), which is enough to learn.
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    
    # Train normal RF on SMOTE data
    rf_sm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sm.fit(X_sm, y_sm)
    results.append(evaluate_technique(rf_sm, X_test, y_test, 'SMOTE (0.1)'))
    
    # ---------------------------------------------------------
    # Technique 2: Class Weights
    # ---------------------------------------------------------
    print("[Technique 2] Class Weights...")
    # WHAT: Set manual weights.
    # ARGUMENT: {0:1, 1:100} -> Error on Class 1 costs 100x more than Class 0.
    w = {0:1, 1:100} 
    rf_w = RandomForestClassifier(n_estimators=100, class_weight=w, random_state=42, n_jobs=-1)
    rf_w.fit(X_train, y_train)
    results.append(evaluate_technique(rf_w, X_test, y_test, f'Class Weight {w}'))

    # ---------------------------------------------------------
    # Technique 3: Threshold Optimization
    # ---------------------------------------------------------
    print("[Technique 3] Threshold Optimization...")
    # WHAT: Using the Class Weight model (rf_w), we scan different thresholds.
    # GOAL: Find threshold that guarantees >= 80% Recall.
    y_prob = rf_w.predict_proba(X_test)[:, 1]
    
    optimal_thresh = 0.5
    
    # LOOP: Check 0.1, 0.15, 0.2 ... 0.9
    for t in np.arange(0.1, 0.9, 0.05):
        y_p = (y_prob >= t).astype(int)
        rec = recall_score(y_test, y_p)
        
        # LOGIC:
        # As threshold goes UP, Recall goes DOWN (we are stricter).
        # We start from low threshold (high recall) and move up until recall drops below 0.80.
        if rec >= 0.80:
            optimal_thresh = t
        else:
            break 
            
    print(f"-> Found Optimal Threshold: {optimal_thresh:.2f}")
    results.append(evaluate_technique(rf_w, X_test, y_test, 'Threshold Tuning', optimal_thresh))
    
    # ---------------------------------------------------------
    # Technique 4: Balanced Random Forest
    # ---------------------------------------------------------
    print("[Technique 4] Balanced Random Forest...")
    # WHAT: Advanced Ensemble.
    # HOW: For each tree, it undersamples the majority class to match the minority size.
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    brf.fit(X_train, y_train)
    results.append(evaluate_technique(brf, X_test, y_test, 'Balanced RF'))

    # Compare All
    print("\n--- Imbalance Technique Comparison ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_imbalance_handling()
