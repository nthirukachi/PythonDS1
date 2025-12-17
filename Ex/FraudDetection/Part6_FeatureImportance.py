"""
Part6_FeatureImportance.py
-------------------------------------------------------------------------------
Part 6: Feature Importance and Interpretability

PROBLEM STATEMENT:
In Finance, we cannot have a "Black Box" model that says "Fraud" without reason.
We must Explain the decisions to analysts.
1. Which features serve as global drivers? (Feature Importance)
2. Why was THIS specific transaction flagged? (SHAP Values)

STEPS TO SOLVE:
1. Train a model.
2. Extract `feature_importances_` (Gini Impurity reduction).
3. Calculate SHAP values (Game Theory contribution).

CONCEPTS & ARGUMENTS:
1. Gini Importance: Fast but biased towards high-cardinality features.
2. SHAP (SHapley Additive exPlanations): State-of-the-art interpretability.
   Values sum up to the prediction probability.
-------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Part2_DataPrep import prepare_data

# NOTE: Handling optional SHAP dependency to ensure code robustness.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP library not found. Skipping SHAP analysis part.")
    SHAP_AVAILABLE = False

def run_feature_importance():
    print("\n=== Part 6: Feature Importance ===")
    
    data = prepare_data()
    if data is None: return
    X_train, _, X_test, y_train, _, _, _ = data
    
    # Train a model for interpretation
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 1. Global Feature Importance (Bar Plot)
    # ---------------------------------------------------------
    print("[Step 1] Plotting Top 15 Important Features...")
    # WHAT: rf.feature_importances_ array (sums to 1).
    importances = rf.feature_importances_
    
    # WHAT: Sort indices largest to smallest.
    indices = np.argsort(importances)[::-1][:15] 
    
    # VISUALIZATION: Bar Chart
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Predictors of Fraud")
    plt.bar(range(15), importances[indices], align="center")
    plt.xticks(range(15), X_train.columns[indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 2. SHAP Values (Local Interpretability)
    # ---------------------------------------------------------
    if SHAP_AVAILABLE:
        print("[Step 2] Calculating SHAP values (using 100 samples)...")
        # WHAT: TreeExplainer is optimized for Tree models.
        explainer = shap.TreeExplainer(rf)
        
        # WHAT: Calculate SHAP values for first 100 test samples.
        # WHY: Full dataset takes too long.
        shap_values = explainer.shap_values(X_test.iloc[:100])
        
        # VISUALIZATION: Summary Plot
        # WHAT: Bee-swarm plot showing range of effects.
        # ARGS: shap_values[1] -> Explaining the positive class (Fraud).
        print("Displaying SHAP Summary Plot...")
        shap.summary_plot(shap_values[1], X_test.iloc[:100])
    
if __name__ == "__main__":
    run_feature_importance()
