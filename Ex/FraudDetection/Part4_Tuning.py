"""
Part4_Tuning.py
-------------------------------------------------------------------------------
Part 4: Hyperparameter Tuning for Best Algorithm

PROBLEM STATEMENT:
Random Forest is the best algorithm (high accuracy, fast prediction).
However, default parameters are rarely optimal.
We aim to maximize **Recall** (catching frauds) by tuning parameters.

STEPS TO SOLVE:
1. Load data.
2. Define a "Grid" of parameters (n_estimators, max_depth, etc).
3. Use `GridSearchCV` to test every combination.
4. Configure scorer to `recall_score` (instead of default accuracy).
5. Run on a 50% subsample (for training speed).

CONCEPTS & ARGUMENTS:
1. GridSearchCV: Brute-force checking of param combinations.
2. n_estimators=[100, 200]: Checking if more trees help.
3. make_scorer(recall_score): Critical. Tells grid search "Best model = Highest Recall".
-------------------------------------------------------------------------------
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from Part2_DataPrep import prepare_data

def run_tuning():
    print("\n=== Part 4: Hyperparameter Tuning ===")
    
    data = prepare_data()
    if data is None: return
    X_train, _, X_test, y_train, _, y_test, _ = data
    
    # ---------------------------------------------------------
    # 1. Subsampling for Speed
    # ---------------------------------------------------------
    # WHAT: Take first 50% of the training data.
    # WHY: Tuning trains the model ~12 times (4 combos * 3 folds).
    #      Training on full data would take 12x longer. 50% is a good trade-off.
    limit = int(len(X_train) * 0.5)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    print(f"[Setup] Tuning on subset of {limit} samples...")
    
    # ---------------------------------------------------------
    # 2. Define Param Grid
    # ---------------------------------------------------------
    # WHAT: Dictionary of settings to explore.
    param_grid = {
        'n_estimators': [100, 200],                  # Try 100 vs 200 trees
        'max_depth': [10, 20],                       # Try medium vs deep trees
        'class_weight': ['balanced', {0:1, 1:50}]    # Try auto-balance vs manual 1:50 weight
    }
    
    # ---------------------------------------------------------
    # 3. Custom Scorer (Maximize Recall)
    # ---------------------------------------------------------
    # WHAT: Create a scoring object for GridSearchCV.
    # ARGUMENTS: 
    # - pos_label=1: We care about Recall of Class 1 (Fraud), not Class 0.
    scorer = make_scorer(recall_score, pos_label=1)
    
    # ---------------------------------------------------------
    # 4. Grid Search Execution
    # ---------------------------------------------------------
    # WHAT: Initialize Search.
    # ARGUMENTS:
    # - estimator: The base model (RandomForest).
    # - param_grid: The dict above.
    # - scoring: Our custom recall scorer.
    # - cv=3: 3-Fold Cross Validation (Train on 2/3, Test on 1/3).
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        scoring=scorer,
        cv=3,       
        verbose=1   # Prints progress log
    )
    
    print("Starting Grid Search...")
    start = time.time()
    
    # WHAT: Run the search (Fit 12 models).
    gs.fit(X_sub, y_sub)
    print(f"Grid Search Completed in {time.time()-start:.2f}s")
    
    # RESULTS
    print(f"Best Hyperparameters Found: {gs.best_params_}")
    print(f"Best Cross-Validation Recall: {gs.best_score_:.4f}")
    
    # ---------------------------------------------------------
    # 5. Final Confirmation
    # ---------------------------------------------------------
    # WHAT: Test the winner model on the held-out Test Set.
    best_rf = gs.best_estimator_
    test_recall = recall_score(y_test, best_rf.predict(X_test))
    print(f"Test Set Recall (Best Model): {test_recall:.4f}")

if __name__ == "__main__":
    run_tuning()
