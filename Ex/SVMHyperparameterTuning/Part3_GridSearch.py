"""
Part3_GridSearch.py
Task: Implement GridSearchCV with 3-fold CV.
"""

import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils import load_and_preprocess_data, evaluate_model

def run_grid_search():
    print("\n=== Part 3: Grid Search ===")
    
    data = load_and_preprocess_data()
    if data is None: return
    X_train, _, X_test, y_train, _, y_test = data # Don't need Val set, CV does that.
    
    # WHAT: Subsampling Data (20% of training data).
    # WHY: Grid Search is O(N^2) or worse with SVM training time. Using full data for many combos is slow.
    limit = int(len(X_train) * 0.2)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    print(f"Using subsample of {limit} records for Grid Search.")
    
    # WHAT: Defining the grid.
    # Total combinations: 4 (C) * 4 (gamma) = 16 models.
    # With 3-fold CV: 16 * 3 = 48 training runs.
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }
    
    # WHAT: Setting up GridSearchCV.
    # ARGUMENTS:
    # - cv=3: 3-fold cross-validation.
    # - n_jobs=-1: Use all CPU cores.
    # - verbose=2: Show progress.
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
    
    print("Starting Grid Search...")
    start_time = time.time()
    grid.fit(X_sub, y_sub)
    total_time = time.time() - start_time
    
    print(f"\nGrid Search Completed in {total_time:.2f}s")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")
    
    # WHAT: Evaluation on Test Set (using best model refit on sub-data automatically? No, mostly just best parameters).
    # Actually GridSearchCV refits on the whole input (X_sub) with best params.
    # To properly evaluate, we should arguably check performance on Test Set.
    best_model = grid.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"Test Set Metrics (Best Model): {test_metrics}")
    
    # WHAT: Printing Results Table.
    results_df = pd.DataFrame(grid.cv_results_)
    # Filtering relevant columns
    cols = ['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']
    print("\nGrid Search Results:")
    print(results_df[cols].sort_values(by='rank_test_score').head(10))
    
    # Return for use in comparison script later?
    return grid.best_params_, test_metrics['Accuracy'], total_time

if __name__ == "__main__":
    run_grid_search()
