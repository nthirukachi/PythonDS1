"""
Part4_Evaluation.py
-------------------------------------------------------------------------------
Part 4: Final Evaluation and Comparison

PROBLEM STATEMENT:
After exploring 3 different strategies (Overfitting, Pre-Pruning, Post-Pruning),
we need to objectively compare them to decide which one is "Production Ready".
We need to compare them not just on Accuracy, but also on Complexity (Depth/Leaves).

STEPS TO SOLVE:
1. Import the runner functions from Parts 1, 2, and 3.
2. Execute each part to get the trained model object.
3. Evaluate each model on the held-out **Test Set** (Data never seen before).
   - This provides an unbiased estimate of real-world performance.
4. Construct a Pandas DataFrame to display a side-by-side comparison table.
5. Analyze the Bias-Variance tradeoff.

EXPECTED OUTPUT:
A Table with columns: [Model, Test Accuracy, Depth, Leaves, Status].
-------------------------------------------------------------------------------
"""

# WHAT: Import pandas library.
# WHY: To create structured tables (DataFrames) for reporting results.
# WHEN: Always used when we need to display data in rows/columns.
import pandas as pd

# WHAT: Import the data loader from our utility script.
# WHY: We need the `X_test` and `y_test` datasets for the final exam.
from utils import load_and_split_data

# WHAT: Import the main execution functions from previous steps.
# WHY: To retrieve the trained model objects (DecisionTreeClassifiers) for comparison.
from Part1_Overfitting import run_part1_overfitting
from Part2_PrePruning import run_part2_prepruning
from Part3_PostPruning import run_part3_postpruning

def run_evaluation():
    """
    Main function to compare all models.
    """
    print("\n=== Part 4: Final Comparison ===")
    
    # 1. Get Test Data
    # WHAT: Call the load function.
    # ARGUMENTS: None.
    # EXPECTED OUTPUT: Unpacking 7 variables. We use `_` for variables we don't need right now (Train/Val sets).
    # We ONLY care about X_test and y_test here.
    _, _, X_test, _, _, y_test, _ = load_and_split_data()
    
    # 2. Get Trained Models
    # WHAT: Run Part 1 to get the Overfit Tree.
    print("\n--- Retrieving Model 1 (Overfit) ---")
    model_overfit = run_part1_overfitting()
    
    # WHAT: Run Part 2 to get the Pre-Pruned Tree (GridSearch Best).
    print("\n--- Retrieving Model 2 (Pre-Pruned) ---")
    model_pre = run_part2_prepruning()
    
    # WHAT: Run Part 3 to get the Post-Pruned Tree (Cost Complexity Best).
    print("\n--- Retrieving Model 3 (Post-Pruned) ---")
    model_post = run_part3_postpruning()
    
    # WHAT: Dictionary organizing our models for iteration.
    models = {
        'Overfit (Baseline)': model_overfit,
        'Pre-Pruned (Grid)': model_pre,
        'Post-Pruned (Alpha)': model_post
    }
    
    results = []
    
    # 3. Evaluation Loop
    print("\n--- Calculating Final Test Metrics ---")
    
    # WHAT: Loop through each model name and object.
    for name, clf in models.items():
        # WHAT: Calculate Accuracy on TEST set.
        # METHOD: .score(X, y). Returns correct_predictions / total_samples.
        # WHY: This is the critical "Unbiased" metric.
        test_acc = clf.score(X_test, y_test)
        
        # WHAT: Get tree structure stats.
        # METHOD: .get_depth() -> How many levels deep is the tree?
        # METHOD: .get_n_leaves() -> How many final decision buckets?
        depth = clf.get_depth()
        leaves = clf.get_n_leaves()
        
        # WHAT: Append a result dictionary to our list.
        # WHY: List of Dicts is the standard way to build a DataFrame row-by-row.
        results.append({
            'Model': name,
            'Test Accuracy': test_acc,
            'Depth': depth,
            'Leaves': leaves,
            # WHAT: Conditional logic for status.
            # LOGIC: If accuracy > 90%, mark as BEST, else POOR.
            'Status': "BEST" if test_acc > 0.9 else "POOR"
        })
    
    # 4. Create Comparison DataFrame
    # WHAT: Convert list of dicts to DataFrame.
    # EXPECTED OUTPUT: A nice printed table.
    df_res = pd.DataFrame(results)
    
    print("\n--- Final Comparison Table (Test Set) ---")
    print(df_res)
    
    # 5. Final Conclusions (Printed to console)
    print("\nFINAL CONCLUSION:")
    print("1. Complexity vs Performance: The Overfit model is unnecessarily complex (High Detph/Leaves).")
    print("2. Efficiency: Post-Pruning often finds the simplest possible tree (Fewest Leaves) that still works well.")
    print("3. Recommendation: Use Post-Pruning (Part 3) for the final production model to minimize overfitting risks.")

if __name__ == "__main__":
    # WHAT: Entry point ensuring this runs only when executed directly.
    run_evaluation()
