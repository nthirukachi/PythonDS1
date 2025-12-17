"""
Part3_PostPruning.py
-------------------------------------------------------------------------------
Part 3: Post-Pruning (Cost Complexity Pruning)

PROBLEM STATEMENT:
Pre-pruning (fixing depth) is rigid. It might stop too early (Underfitting).
"Post-Pruning" grows the FULL tree first, then snips off branches that don't add
enough value relative to their complexity.

STEPS TO SOLVE:
1. Calculate the `pruning_path` (Sequence of `ccp_alpha` values).
2. Train a tree for every alpha value.
3. Pick the alpha that gives the highest Validation Accuracy.

CONCEPTS & ARGUMENTS:
- ccp_alpha (Cost Complexity Parameter): A penalty term for tree size.
  - alpha=0: Full Overfit tree.
  - alpha=High: Tree is just a single root node (Underfit).
  - We look for the "Sweet Spot".
-------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import load_and_split_data

def run_part3_postpruning():
    print("\n=== Part 3: Post-Pruning (Cost Complexity) ===")
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split_data()
    
    # ---------------------------------------------------------
    # 1. Get Pruning Path
    # ---------------------------------------------------------
    # WHAT: Sklearn calculates the "weakest link" alphas for us.
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    print(f"Found {len(ccp_alphas)} candidate alphas.")
    
    # ---------------------------------------------------------
    # 2. Train Models for each Alpha
    # ---------------------------------------------------------
    clfs = []
    train_scores = []
    val_scores = []
    
    for ccp_alpha in ccp_alphas:
        # Train new tree with this specific penalty
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        val_scores.append(clf.score(X_val, y_val))
    
    # ---------------------------------------------------------
    # 3. Find Best Alpha
    # ---------------------------------------------------------
    # WHAT: Find index where Validation Score is max.
    best_idx = val_scores.index(max(val_scores))
    best_alpha = ccp_alphas[best_idx]
    best_clf = clfs[best_idx]
    
    print(f"Optimal Alpha: {best_alpha:.5f}")
    print(f"Best Val Score: {val_scores[best_idx]:.4f}")
    print(f"Tree Size (Leaves): {best_clf.get_n_leaves()}")
    
    # ---------------------------------------------------------
    # 4. Plot Alpha vs Accuracy
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, train_scores, marker='o', label="Train", drawstyle="steps-post")
    plt.plot(ccp_alphas, val_scores, marker='o', label="Validation", drawstyle="steps-post")
    plt.xlabel("alpha (Complexity Penalty)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Alpha")
    plt.legend()
    plt.grid(True)
    
    plt.savefig('PostPruning_AlphaCurve.png')
    print("PostPruning_AlphaCurve.png saved.")
    plt.close()
    
    return best_clf

if __name__ == "__main__":
    run_part3_postpruning()
