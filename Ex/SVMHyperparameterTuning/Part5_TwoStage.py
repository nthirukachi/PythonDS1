"""
Part5_TwoStage.py
Task: Two-Stage Coarse-to-Fine Strategy (Random Search -> Grid Search).
"""

import time
import numpy as np
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from utils import load_and_preprocess_data, evaluate_model

def run_two_stage():
    print("\n=== Part 5: Two-Stage Strategy ===")
    
    data = load_and_preprocess_data()
    if data is None: return
    X_train, _, X_test, y_train, _, y_test = data
    
    start_total = time.time()
    
    # ----------------- STAGE 1: COARSE RANDOM SEARCH (10% Data) -----------------
    print("\n--- Stage 1: Coarse Random Search (10% Data) ---")
    limit_1 = int(len(X_train) * 0.1)
    X_sub1 = X_train[:limit_1]
    y_sub1 = y_train[:limit_1]
    
    # Wide Search Space
    param_dist = {
        'C': loguniform(0.01, 1000),
        'gamma': loguniform(0.0001, 10)
    }
    
    # Fewer iterations (15) to cast a wide net quickly
    rs = RandomizedSearchCV(SVC(random_state=42), param_dist, n_iter=15, cv=3, n_jobs=-1, random_state=42)
    rs.fit(X_sub1, y_sub1)
    
    best_coarse = rs.best_params_
    print(f"Stage 1 Best Params: {best_coarse}")
    
    # ----------------- STAGE 2: FINE GRID SEARCH (50% Data) -----------------
    print("\n--- Stage 2: Fine Grid Search (50% Data) ---")
    limit_2 = int(len(X_train) * 0.5)
    X_sub2 = X_train[:limit_2]
    y_sub2 = y_train[:limit_2]
    
    # Define narrow grid around Stage 1 results (e.g., 0.5x, 1x, 2x)
    # WHAT: Creating a small grid centered on the winner of Stage 1.
    c_center = best_coarse['C']
    g_center = best_coarse['gamma']
    
    # Generating 3 values: half, same, double (approx)
    fine_grid = {
        'C': [c_center * 0.5, c_center, c_center * 2],
        'gamma': [g_center * 0.5, g_center, g_center * 2]
    }
    
    print(f"Fine Tuning Grid: {fine_grid}")
    
    gs = GridSearchCV(SVC(random_state=42), fine_grid, cv=3, n_jobs=-1)
    gs.fit(X_sub2, y_sub2)
    
    total_time = time.time() - start_total
    
    print(f"\nTwo-Stage Completed in {total_time:.2f}s")
    print(f"Final Best Params: {gs.best_params_}")
    
    # Evaluation
    best_model = gs.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"Test Set Metrics: {test_metrics}")
    
    return gs.best_params_, test_metrics['Accuracy'], total_time

if __name__ == "__main__":
    run_two_stage()
