"""
Part4_RandomSearch.py
Task: Implement Random Search with RandomizedSearchCV.
"""

import time
import pandas as pd
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from utils import load_and_preprocess_data, evaluate_model

def run_random_search():
    print("\n=== Part 4: Random Search ===")
    
    data = load_and_preprocess_data()
    if data is None: return
    X_train, _, X_test, y_train, _, y_test = data
    
    # WHAT: Subsampling (20%).
    limit = int(len(X_train) * 0.2)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # WHAT: Defining Parameter Distributions.
    # loguniform: Samples evenly across orders of magnitude (e.g., probability of picking 0.01-0.1 is same as 10-100).
    # WHY: Hyperparameters like C and Gamma span multiple scales. Uniform sampling would Bias towards large numbers.
    param_dist = {
        'C': loguniform(0.01, 100),
        'gamma': loguniform(0.0001, 1)
    }
    
    # WHAT: RandomizedSearchCV.
    # ARGUMENTS: n_iter=20. We try 20 random combinations.
    random_search = RandomizedSearchCV(SVC(random_state=42), param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=1, random_state=42)
    
    print("Starting Random Search...")
    start_time = time.time()
    random_search.fit(X_sub, y_sub)
    total_time = time.time() - start_time
    
    print(f"\nRandom Search Completed in {total_time:.2f}s")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_:.4f}")
    
    # Evaluation
    best_model = random_search.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"Test Set Metrics: {test_metrics}")
    
    return random_search.best_params_, test_metrics['Accuracy'], total_time

if __name__ == "__main__":
    run_random_search()
