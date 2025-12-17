"""
Part3_Tuning.py
-------------------------------------------------------------------------------
Part 3: Hyperparameter Tuning

PROBLEM STATEMENT:
Models have "Knobs" (Hyperparameters) like `n_estimators` (number of trees) or 
`max_depth`. The default settings in Scikit-Learn are good, but not optimal.
Manually guessing these is inefficient.
We need an automated way to search for the best combination.

STEPS TO SOLVE:
1. Define a "Search Space" (Range of values for each parameter).
2. Use **RandomizedSearchCV** to randomly sample 10 combinations.
3. Train and Evaluate each.
4. Report the best parameters found.

EXPECTED OUTPUT:
- Best Parameter Dictionary (e.g. {'n_estimators': 200, ...}).
- Improved Test Accuracy.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Import RF Class.
from sklearn.ensemble import RandomForestClassifier

# WHAT: Import RandomizedSearchCV.
# WHY: GridSearch tests ALL combinations (Slow). RandomSearch tests a few random ones (Fast).
# Often finds a result 95% as good in 5% of the time.
from sklearn.model_selection import RandomizedSearchCV

from utils import load_loan_data

def run_part3_tuning():
    print("\n=== Part 3: Hyperparameter Tuning ===")
    
    X_train, X_test, y_train, y_test, _ = load_loan_data()
    
    # ---------------------------------------------------------
    # 1. Define Search Space
    # ---------------------------------------------------------
    # WHAT: A dictionary defining distributions.
    # Lists [] mean "pick one of these".
    param_dist = {
        # How many trees? More is better but slower.
        'n_estimators': [50, 100, 200],         
        
        # How complex can each tree be?
        # None = Infinite depth (Risk of Overfitting).
        # 5 = Very simple (Risk of Underfitting).
        'max_depth': [5, 10, 15, 20, None],     
        
        # Minimum people in a leaf.
        # 1 = Sensitive to noise. 10 = Very smooth.
        'min_samples_leaf': [1, 5, 10]          
    }
    
    print(f"Tuning Hyperparameters over space: {param_dist}")
    
    # ---------------------------------------------------------
    # 2. Randomized Search
    # ---------------------------------------------------------
    # WHAT: Initialize the Search Object.
    # ARGUMENTS:
    # - estimator: The model template (RF).
    # - param_distributions: The dictionary above.
    # - n_iter=10: "Try 10 different random combinations".
    # - cv=3: "Use 3-Fold Cross Validation for each try".
    # - n_jobs=-1: Parallelize.
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10, 
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    print("Running Randomized Search (Training 10 candidates x 3 folds = 30 fits)...")
    
    # WHAT: Run the search. This takes time.
    rs.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 3. Results
    # ---------------------------------------------------------
    # WHAT: Print the winner.
    print(f"Best Parameters Found: {rs.best_params_}")
    print(f"Best Validation Accuracy: {rs.best_score_:.4f}")
    
    # WHAT: Evaluate the Winner on the Held-Out Test Set.
    # .best_estimator_ automatically gives us the model trained with the best params.
    best_model = rs.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    
    print(f"Final Test Set Accuracy (Best Model): {test_acc:.4f}")
    
    return best_model

if __name__ == "__main__":
    run_part3_tuning()
