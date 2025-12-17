"""
Part1_Overfitting.py
-------------------------------------------------------------------------------
Part 1: Demonstrate Overfitting in Decision Trees

PROBLEM STATEMENT:
Decision Trees have a tendency to "Overfit" if left unchecked.
They will grow until every single leaf is pure (contains only 1 class).
This results in 100% Training Accuracy but poor performance on new data (Validation).

STEPS TO SOLVE:
1. Load Data (Train/Val/Test).
2. Train a Decision Tree with NO constraints (default settings).
3. Measure Accuracy on Train vs Validation.
4. Visualize the tree structure (It will be huge).
5. Generate a "Learning Curve" by manually limiting depth from 1 to 20 to see the gap widen.

EXPECTED OUTPUT:
- Train Accuracy: 1.0 (or very close).
- Val Accuracy: Lower (e.g., 0.91).
- A large gap indicating Overfitting.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from utils import load_and_split_data, plot_learning_curve

def run_part1_overfitting():
    print("\n=== Part 1: Demonstate Overfitting ===")
    
    # 1. Load Data
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = load_and_split_data()
    
    # ---------------------------------------------------------
    # 2. Train UNCONSTRAINED Tree
    # ---------------------------------------------------------
    print("Training Unconstrained Decision Tree...")
    # ARGUMENTS:
    # - random_state=42: Deterministic.
    # - max_depth=None: (Default) Allow tree to grow infinitely deep.
    dt_overfit = DecisionTreeClassifier(random_state=42)
    dt_overfit.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 3. Measure Stats
    # ---------------------------------------------------------
    train_acc = dt_overfit.score(X_train, y_train)
    val_acc = dt_overfit.score(X_val, y_val)
    
    # WHAT: Get tree properties.
    depth = dt_overfit.get_depth()
    n_leaves = dt_overfit.get_n_leaves()
    
    print(f"Overfit Model Stats:")
    print(f" - Train Acc: {train_acc:.4f} (Perfect Memorization)")
    print(f" - Val Acc:   {val_acc:.4f}")
    print(f" - Depth:     {depth}")
    print(f" - Leaves:    {n_leaves}")
    
    # ---------------------------------------------------------
    # 4. Generate Learning Curve (Depth 1 to 20)
    # ---------------------------------------------------------
    print("Generating Learning Curve (Depth vs Accuracy)...")
    depths = range(1, 21)
    train_scores = []
    val_scores = []
    
    for d in depths:
        # WHAT: Train a new tree restricted to depth 'd'.
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        
        train_scores.append(clf.score(X_train, y_train))
        val_scores.append(clf.score(X_val, y_val))
        
    # Plot using helper
    plot_learning_curve(depths, train_scores, val_scores, title="Overfitting_Curve_Depth")
    
    return dt_overfit

if __name__ == "__main__":
    run_part1_overfitting()
