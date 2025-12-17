"""
Part2_Interpretation.py
-------------------------------------------------------------------------------
Part 2: Feature Importance & Interpretability (SHAP)

PROBLEM STATEMENT:
Machine Learning models are often "Black Boxes".
A Random Forest might have 76% accuracy, but a bank cannot legally reject a loan
without giving a REASON (e.g., "Insufficient Income").
We need techniques to look inside the box.

STEPS TO SOLVE:
1. Reuse the trained RF model from Part 1.
2. Global Explanation: Extract `feature_importances_` to see what matters generally.
3. Local Explanation: Use `SHAP` (Shapley Additive exPlanations) to explain 
   specific individual predictions.

EXPECTED OUTPUT:
- A Bar Chart of Top 10 Features.
- A SHAP Summary plot.
- A textual explanation of why a specific high-risk customer was rejected.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# WHAT: Import SHAP library.
# WHY: SHAP provides consistent, game-theoretic feature attribution.
# It is the gold standard for interpreting ensemble tree models.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    # WHAT: Fallback if library is missing.
    SHAP_AVAILABLE = False
    print("WARNING: 'shap' library not found. SHAP plots will be skipped.")

# WHAT: Import function to get the model.
from Part1_RF_vs_Tree import run_part1_comparison

def run_part2_interpretation():
    print("\n=== Part 2: Feature Importance ===")
    
    # 1. Get Trained Model
    # WHAT: Load the model and data from Part1 (so we don't have to retrain).
    rf_model, X_train, X_test, y_test = run_part1_comparison()
    
    # WHAT: Get column names for labeling plots.
    feature_names = X_test.columns
    
    # ---------------------------------------------------------
    # 2. Global Feature Importance (MDI)
    # ---------------------------------------------------------
    print("Calculating Feature Importance...")
    
    # WHAT: Extract Importance Scores.
    # PROPERTY: .feature_importances_
    # DEFINITION: How much does each feature decrease "Impurity" (Gini) across all trees?
    importances = rf_model.feature_importances_
    
    # WHAT: Sort indices to find Top 10.
    # .argsort() returns sorted indices. [::-1] reverses it (High to Low).
    indices = np.argsort(importances)[::-1][:10]
    
    # WHAT: Plot Bar Chart.
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.bar(range(10), importances[indices], align="center")
    
    # Label x-axis with feature names.
    plt.xticks(range(10), feature_names[indices], rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('RF_Feature_Importance.png')
    print("Saved RF_Feature_Importance.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 3. SHAP Explanation (Local Interpretability)
    # ---------------------------------------------------------
    if SHAP_AVAILABLE:
        print("\nCalculating SHAP Values (Local Interpretability)...")
        
        # WHAT: Initialize Explainer.
        # ARGUMENT: rf_model. passed to TreeExplainer which parses the internal trees.
        explainer = shap.TreeExplainer(rf_model)
        
        # WHAT: Calculate Shapley values for a subset of Test Data.
        # WHY: Calculating for all 10k rows is slow. 100 is enough for a summary.
        # X_test.iloc[:100]: First 100 rows.
        X_sample = X_test.iloc[:100]
        shap_values = explainer.shap_values(X_sample)
        
        # WHAT: Handle SHAP Output format.
        # RF Classifier output is a list of [Values_Class0, Values_Class1].
        # We care about Class 1 (Default Risk).
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values
            
        # PLOT: Summary Plot (Beeswarm).
        print("Generating SHAP Summary Plot...")
        plt.figure()
        # WHAT: Generates a dot-plot showing feature impact (x-axis) vs feature value (color).
        shap.summary_plot(shap_values_class1, X_sample, show=False)
        plt.savefig('SHAP_Summary.png')
        print("Saved SHAP_Summary.png")
        plt.close()
        
        # ---------------------------------------------------------
        # 4. Explain a Specific Prediction
        # ---------------------------------------------------------
        print("Generating Explanation for a Specific High-Risk Customer...")
        
        # WHAT: Predict probabilities for entire test set.
        # [:, 1] gets the probability of Class 1 (Default).
        probs = rf_model.predict_proba(X_test)[:, 1]
        
        # WHAT: Find the person with the HIGHEST risk score.
        risky_idx = np.argmax(probs)
        
        print(f"Analyzing Customer Index {risky_idx}...")
        print(f"Predicted Probability of Default: {probs[risky_idx]:.2f}")
        
        # WHAT: Get SHAP values for this specific person.
        # This vector tells us how much each feature pushed the probability UP or DOWN.
        # If shap_values is list (Binary Classification), grab class 1 index.
        # If we computed for sample only, we can't use risky_idx directly if it's outside sample.
        # Let's re-compute for this specific person to be safe.
        person_X = X_test.iloc[[risky_idx]] # Double brackets to keep it a DataFrame
        person_shap_values = explainer.shap_values(person_X)
        
        if isinstance(person_shap_values, list):
            sv = person_shap_values[1][0] # Class 1, Person 0
        else:
            sv = person_shap_values[0]
            
        # WHAT: Explain top contributing factors.
        # Zip Feature Names with their numerical impact.
        impacts = list(zip(feature_names, sv))
        
        # Sort by magnitude (Absolute impact).
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 3 Reasons for Rejection:")
        for feat, score in impacts[:3]:
            # Positive Score = Pushed towards Default (1).
            # Negative Score = Pushed towards Good (0).
            direction = "INCREASED risk" if score > 0 else "DECREASED risk"
            print(f" - {feat}: {direction} (Impact: {score:.3f})")

if __name__ == "__main__":
    run_part2_interpretation()
