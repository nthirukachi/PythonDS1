"""
Part2_PrePruning.py
-------------------------------------------------------------------------------
Part 2: Implement Pre-Pruning (Hyperparameter Tuning)

PROBLEM STATEMENT:
Since Unconstrained trees overfit (Part 1), we must apply "Brakes" to stops growth.
This is called **Pre-Pruning**.
We constrain the tree's geometry using Hyperparameters like:
- `max_depth`: Max levels allowed.
- `min_samples_leaf`: Min data points required to form a leaf.

STEPS TO SOLVE:
1. Load Data.
2. Define a "Grid" of possible parameter combinations (Hypothesis Space).
3. Run `GridSearchCV` to test ALL combinations using Cross-Validation.
4. Visualize performance as a Heatmap.

EXPECTED OUTPUT:
- Best Parameter Set (e.g. Depth=3, Leaf=5).
- A Heatmap PNG showing where accuracy is highest.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# WHAT: Import classifier and search tool.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from utils import load_and_split_data

def run_part2_prepruning():
    print("\n=== Part 2: Pre-Pruning (Grid Search) ===")
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split_data()
    
    # ---------------------------------------------------------
    # 1. Define Parameter Grid
    # ---------------------------------------------------------
    # WHAT: A dictionary listing the options we want to explore.
    param_grid = {
        # EXPERIMENT: Try limiting depth to 3, 5, 10, or None (Infinite).
        'max_depth': [3, 5, 10, None],
        
        # EXPERIMENT: Try forcing leaves to contain at least 1, 5, 10, or 20 samples.
        # WHY: Larger leaf size = Smoother boundaries = Less Overfitting.
        'min_samples_leaf': [1, 5, 10, 20]
    }
    
    print(f"Searching parameters: {param_grid}")
    
    # ---------------------------------------------------------
    # 2. Execute Grid Search
    # ---------------------------------------------------------
    # WHAT: Initialize GridSearchCV object.
    # ARGUMENTS:
    # - estimator: The model type (DecisionTree).
    # - param_grid: The options we defined above.
    # - cv=3: 3-Fold Cross Validation. Splits Train data into 3 chunks to validte internally.
    # - scoring='accuracy': We want to maximize Accuracy.
    # - return_train_score=True: Useful to check for overfitting during search.
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=3,
        return_train_score=True,
        scoring='accuracy'
    )
    
    # WHAT: Run the brute-force search.
    # This trains 4x4=16 combinations * 3 folds = 48 total models.
    gs.fit(X_train, y_train)
    
    # WHAT: Print the winner.
    # attribute .best_params_: Returns dictionary of optimal settings.
    # attribute .best_score_: Returns the average accuracy of that setting.
    print(f"Best Params Found: {gs.best_params_}")
    print(f"Best Cross-Validation Score: {gs.best_score_:.4f}")
    
    # ---------------------------------------------------------
    # 3. Visualize Heatmap
    # ---------------------------------------------------------
    # WHAT: Extract search results into a detailed DataFrame.
    results = pd.DataFrame(gs.cv_results_)
    
    # WHAT: Preprocessing for Plotting.
    # Convert 'None' (Infinite Depth) to string "None" so it can be plotted as a label.
    results['param_max_depth'] = results['param_max_depth'].fillna('None')
    
    # WHAT: Create Pivot Table (Matrix format).
    # Rows=Depth, Cols=LeafSize, Values=Accuracy.
    pivoted = results.pivot(index='param_max_depth', columns='param_min_samples_leaf', values='mean_test_score')
    
    # WHAT: Draw Heatmap.
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivoted, annot=True, cmap='viridis', fmt='.3f')
    
    plt.title("Val Accuracy: Depth vs Min Samples Leaf")
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("Max Depth")
    
    # WHAT: Save plot.
    plt.savefig('PrePruning_Heatmap.png')
    print("PrePruning_Heatmap.png saved.")
    plt.close()
    
    # Return the best model ready for final testing
    return gs.best_estimator_

if __name__ == "__main__":
    run_part2_prepruning()
