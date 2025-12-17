"""
Part1_Overfitting.py
-------------------------------------------------------------------------------
Part 1: Demonstrate Overfitting in Decision Trees

PROBLEM STATEMENT:
Decision Trees are "Non-Parametric" models, meaning they can grow infinitely complex.
If not constrained, a Tree will keep splitting until every single training example
is correctly classified (Accuracy = 100%).
While this sounds good, it means the model is memorizing "Noise" instead of patterns.
It will fail when seeing new data (Validation Set).

STEPS TO SOLVE:
1. Load the 70/15/15 split data.
2. Train a `DecisionTreeClassifier` with `max_depth=None` (Unlimited).
3. Observe the "Gap" between Perfect Training Score and Lower Validation Score.
4. Visualize the "Learning Curve" by training trees of increasing depths (1 to 20).

EXPECTED OUTPUT:
- A plot showing Training Accuracy sticking to 1.0, while Validation Accuracy drops.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Import plotting library.
import matplotlib.pyplot as plt

# WHAT: Import the Tree Classifier.
# WHY: The core algorithm we are studying.
from sklearn.tree import DecisionTreeClassifier

# WHAT: Import accuracy metric.
# WHY: To measure percentage of correct predictions.
from sklearn.metrics import accuracy_score

# WHAT: Import our custom helper functions.
# WHY: Reusing code from utils.py keeps this file clean.
from utils import load_and_split_data, plot_learning_curve

def run_part1_overfitting():
    print("\n=== Part 1: Demonstrate Overfitting ===")
    
    # 1. Load Data
    # WHAT: Unpack the data variables returned by utils.
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = load_and_split_data()
    
    # ---------------------------------------------------------
    # 2. Train UNCONSTRAINED Tree
    # ---------------------------------------------------------
    print("Training Unconstrained Decision Tree...")
    
    # WHAT: Initialize the Model.
    # ARGUMENTS:
    # - random_state=42: Ensures consistent results.
    # - max_depth=None: (Default) This is the DANGER setting. It tells the tree:
    #   "Keep splitting until leaves are pure". This causes Overfitting.
    dt_overfit = DecisionTreeClassifier(random_state=42, max_depth=None)
    
    # WHAT: Train (Fit) the model.
    # WHY: This builds the tree structure based on X_train patterns.
    dt_overfit.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 3. Measure Stats
    # ---------------------------------------------------------
    # WHAT: Score on Training Data.
    # EXPECTED: Close to 1.0 (100%).
    train_acc = dt_overfit.score(X_train, y_train)
    
    # WHAT: Score on Validation Data (New Data).
    # EXPECTED: Significantly lower than Train Acc (e.g. 0.93 vs 1.0).
    val_acc = dt_overfit.score(X_val, y_val)
    
    # WHAT: Get physical properties of the learned tree.
    # Method .get_depth(): Height of the tree (Root to deepeset leaf).
    # Method .get_n_leaves(): Total number of endpoints.
    depth = dt_overfit.get_depth()
    n_leaves = dt_overfit.get_n_leaves()
    
    print(f"Overfit Model Stats:")
    print(f" - Train Acc: {train_acc:.4f} (Perfect Memorization)")
    print(f" - Val Acc:   {val_acc:.4f} (Generalization Performance)")
    print(f" - Depth:     {depth} (Likely very deep)")
    print(f" - Leaves:    {n_leaves} (High complexity)")
    
    # ---------------------------------------------------------
    # 4. Generate Learning Curve
    # ---------------------------------------------------------
    print("Generating Learning Curve (Depth 1 to 20)...")
    
    # WHAT: Define range of depths to test.
    # range(1, 21): generates numbers 1, 2, ... 20.
    depths = range(1, 21)
    train_scores = []
    val_scores = []
    
    # WHAT: Loop through each depth limit.
    for d in depths:
        # WHAT: Create a NEW tree with specific constraint 'max_depth=d'.
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        
        # WHAT: Train it.
        clf.fit(X_train, y_train)
        
        # WHAT: Record scores.
        train_scores.append(clf.score(X_train, y_train))
        val_scores.append(clf.score(X_val, y_val))
        
    # WHAT: Call helper to draw the plot.
    # EXPECTED OUTPUT: A PNG file named 'Overfitting_Curve_Depth.png'.
    plot_learning_curve(depths, train_scores, val_scores, title="Overfitting_Curve_Depth")
    
    return dt_overfit

if __name__ == "__main__":
    run_part1_overfitting()
