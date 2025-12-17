"""
Part2_PrePruning.py
-------------------------------------------------------------------------------
Part 2: Implement Pre-Pruning (Hyperparameter Tuning)

PROBLEM STATEMENT:
"Pre-Pruning" means stopping the tree *before* it grows too complex.
We do this by setting constraints like `max_depth` or `min_samples_split`.

STEPS TO SOLVE:
1. Define a "Grid" of parameters to explore.
2. Use `GridSearchCV` to test every combination on the Training Set.
3. Validate on the Validation Set.
4. Visualize the results (Heatmap).

CONCEPTS & ARGUMENTS:
- max_depth: Hard limit on levels.
- min_samples_split: Don't split a node if it has fewer than X samples (prevents isolating noise).
- min_samples_leaf: Leaf node must have at least X samples.
-------------------------------------------------------------------------------
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import load_and_split_data

def run_part2_prepruning():
    print("\n=== Part 2: Pre-Pruning (Grid Search) ===")
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split_data()
    
    # ---------------------------------------------------------
    # 1. Define Parameter Grid
    # ---------------------------------------------------------
    # WHAT: Dictionary of settings we want to test.
    param_grid = {
        'max_depth': [3, 5, 10, None],          # Constrain height
        'min_samples_leaf': [1, 5, 10, 20]      # Constrain leaf size
    }
    
    print(f"Searching grid: {param_grid}")
    
    # ---------------------------------------------------------
    # 2. Execute Grid Search
    # ---------------------------------------------------------
    # ARGUMENTS:
    # - cv=3: 3-Fold Cross Validation.
    # - return_train_score=True: Key for detecting overfitting Gap.
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=3,
        return_train_score=True,
        scoring='accuracy'
    )
    
    gs.fit(X_train, y_train)
    
    print(f"Best Params: {gs.best_params_}")
    print(f"Best CV Score: {gs.best_score_:.4f}")
    
    # ---------------------------------------------------------
    # 3. Visualize Heatmap
    # ---------------------------------------------------------
    # WHAT: Extract results into a DataFrame.
    results = pd.DataFrame(gs.cv_results_)
    
    # WHAT: Create Pivot Table for Heatmap (Depth vs Leaf Size).
    # Since None is not sortable, we fill it with string 'None'.
    results['param_max_depth'] = results['param_max_depth'].fillna('None')
    pivoted = results.pivot(index='param_max_depth', columns='param_min_samples_leaf', values='mean_test_score')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivoted, annot=True, cmap='viridis', fmt='.3f')
    plt.title("Val Accuracy: Depth vs Min Samples Leaf")
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("Max Depth")
    
    plt.savefig('PrePruning_Heatmap.png')
    print("PrePruning_Heatmap.png saved.")
    plt.close()
    
    return gs.best_estimator_

if __name__ == "__main__":
    run_part2_prepruning()
