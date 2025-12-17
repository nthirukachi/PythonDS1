"""
Part3_PostPruning.py
-------------------------------------------------------------------------------
Part 3: Post-Pruning (Cost Complexity Pruning)

PROBLEM STATEMENT:
Pre-pruning (Part 2) is "Greedy". By setting a hard limit like `max_depth=3`, 
we might miss a very important split that happens at depth 4.
**Post-Pruning** solves this by:
1. Growing the tree to full size (letting it overfit).
2. Pruning it back (cutting branches) based on a mathematical score: **Cost Complexity**.

STEPS TO SOLVE:
1. Generate the "Pruning Path": A list of `alpha` values (Penalties) that trigger cuts.
2. Train a separate Decision Tree for EACH alpha candidate.
3. Compare Validation Accuracy for each tree.
4. Select the Alpha that maximizes Validation Accuracy.

CONCEPTS & ARGUMENTS:
- ccp_alpha: Cost-Complexity Parameter. 
  - If 0, tree is Full (Overfit).
  - If High, tree becomes a stump (Underfit).
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import load_and_split_data

def run_part3_postpruning():
    print("\n=== Part 3: Post-Pruning (Cost Complexity) ===")
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split_data()
    
    # ---------------------------------------------------------
    # 1. Get Pruning Path (Candidates)
    # ---------------------------------------------------------
    print("Calculating Pruning Path...")
    
    # WHAT: Initialize a standard tree.
    clf = DecisionTreeClassifier(random_state=42)
    
    # WHAT: Calculate the path.
    # METHOD: .cost_complexity_pruning_path(X, y).
    # RETURNS: A bunch object containing 'ccp_alphas' and 'impurities'.
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    
    # WHAT: Extract the list of alphas (The penalty thresholds).
    ccp_alphas = path.ccp_alphas
    
    print(f"Found {len(ccp_alphas)} candidate alphas (potential cut variations).")
    
    # ---------------------------------------------------------
    # 2. Train Models for each Alpha
    # ---------------------------------------------------------
    clfs = []
    train_scores = []
    val_scores = []
    
    print("Training trees for each alpha candidate...")
    # WHAT: Loop through every candidate alpha value.
    for ccp_alpha in ccp_alphas:
        # WHAT: Create a new tree with this specific `ccp_alpha`.
        # WHY: This forces the tree to simplify itself according to the penalty.
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        
        # WHAT: Train it.
        clf.fit(X_train, y_train)
        
        # WHAT: Store model and scores.
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        val_scores.append(clf.score(X_val, y_val))
    
    # ---------------------------------------------------------
    # 3. Find Best Alpha (Optimization)
    # ---------------------------------------------------------
    # WHAT: Find the index where Validation Score matches the maximum.
    # LOGIC: We want the model that performs best on UNSEEN (Validation) data.
    max_score = max(val_scores)
    best_idx = val_scores.index(max_score)
    
    # WHAT: Retrieve the winning settings.
    best_alpha = ccp_alphas[best_idx]
    best_clf = clfs[best_idx]
    
    print(f"Optimal Alpha Selected: {best_alpha:.5f}")
    print(f"Best Validation Score: {val_scores[best_idx]:.4f}")
    print(f"Final Tree Size (Leaves): {best_clf.get_n_leaves()}")
    
    # ---------------------------------------------------------
    # 4. Plot Alpha vs Accuracy Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # WHAT: Plot Train Scores vs Alphas.
    # ARG: drawstyle="steps-post". Pruning happens in discrete steps/jumps.
    plt.plot(ccp_alphas, train_scores, marker='o', label="Train", drawstyle="steps-post")
    
    # WHAT: Plot Validation Scores vs Alphas.
    plt.plot(ccp_alphas, val_scores, marker='o', label="Validation", drawstyle="steps-post")
    
    plt.xlabel("alpha (Complexity Penalty)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Alpha (Selecting the Peak)")
    plt.legend()
    plt.grid(True)
    
    # WHAT: Save plot.
    plt.savefig('PostPruning_AlphaCurve.png')
    print("PostPruning_AlphaCurve.png saved.")
    plt.close()
    
    return best_clf

if __name__ == "__main__":
    run_part3_postpruning()
