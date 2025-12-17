"""
UseCase3_Health.py
-------------------------------------------------------------------------------
USE CASE 3: Population Health Risk Scoring

PROBLEM STATEMENT:
Real-world medical data is messy. Records often have "Missing Values" (NaNs) due to 
errors, patient refusal, or machine failure.
Most algorithms (Logistic Regression, standard SVM) crash if fed NaNs.
We need to handle this robustly.

STEPS TO SOLVE:
1. Load Data and naturally inject 10% Missing Values (NaN).
2. Strategy A: Use an **Imputation Pipeline** (Fill NaNs with Average -> Train RF).
3. Strategy B: Use **Native Handling** (HistGradientBoosting).
   - Some modern trees can learn "Missing" as a piece of information itself.
4. Compare performance (R2 Score).

CONCEPTS & ARGUMENTS:
1. Imputation: Guessing the missing value based on the column mean/median.
2. HistGradientBoostingRegressor: A boosted tree algorithm (like XGBoost/LightGBM) 
   that efficiently handles NaNs without needing imputation.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# WHAT: Import Regressors (Predicting a score, not a class).
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# WHAT: Import Pipeline Tools.
# SimpleImputer: Fills missing slots.
# make_pipeline: Chains steps together (Impute -> Scale -> Train).
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  make_pipeline

# WHAT: Metrics.
# R2 Score: How well the regression line fits (1.0 is perfect).
from sklearn.metrics import r2_score
from utils import generate_health_data

def run_health_use_case():
    print("\n=== USE CASE 3: POPULATION HEALTH (MISSING DATA) ===")
    
    # ---------------------------------------------------------
    # 1. Data Loading (With Errors)
    # ---------------------------------------------------------
    # WHAT: Get dataset. X_df contains 10% np.nan values.
    X_df, y = generate_health_data() 
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    results = []
    
    # ---------------------------------------------------------
    # Strategy 1: Classical Imputation + Random Forest
    # ---------------------------------------------------------
    print("Training Random Forest (with Mean Imputation)...")
    
    # WHAT: Create a container that executes steps in order.
    # STEP 1: SimpleImputer(strategy='mean') -> Finds average of column and fills valid.
    # STEP 2: RandomForestRegressor -> Trains on the clean data.
    # WHY: If we ran RF directly on X_train, it would crash.
    pipe_rf = make_pipeline(
        SimpleImputer(strategy='mean'),
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    
    # Fit (Imputes then Trains)
    pipe_rf.fit(X_train, y_train)
    
    # Score
    score_rf = pipe_rf.score(X_test, y_test)
    results.append({'Method': 'RF + Mean Impute', 'R2 Score': score_rf})
    
    # ---------------------------------------------------------
    # Strategy 2: Native Handling (Modern Approach)
    # ---------------------------------------------------------
    print("Training HistGradientBoosting (Native Handling)...")
    
    # WHAT: Initialize HGBR.
    # WHY: This algo supports NaNs natively. It learns: "If value is Missing, go Left".
    # This captures information (e.g., "Missing BP" might mean "Patient too unstable to measure").
    # Imputation destroys this signal by hiding it with the Mean.
    hgb = HistGradientBoostingRegressor(random_state=42)
    
    # Fit (Directly on NaN data)
    hgb.fit(X_train, y_train)
    
    score_hgb = hgb.score(X_test, y_test)
    results.append({'Method': 'HistGradBoost (Native)', 'R2 Score': score_hgb})
    
    print("\nComparison:")
    print(pd.DataFrame(results))
    
    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------
    print("\nFeature Importance (from RF pipeline):")
    # WHAT: Extract the model from the pipeline to see features.
    rf_model = pipe_rf.named_steps['randomforestregressor']
    importances = rf_model.feature_importances_
    
    # Display Top 5
    feat_names = X_df.columns
    top_indices = np.argsort(importances)[::-1][:5]
    print(pd.Series(importances[top_indices], index=feat_names[top_indices]))

    print("\nCONCLUSION FOR USE CASE 3:")
    print("- Native handling (HistGradientBoosting) usually performs better/faster.")
    print("- Imputation is a 'Band-Aid' solution.")
    print("- Missingness itself can be a predictor (informative missingness).")

if __name__ == "__main__":
    run_health_use_case()
