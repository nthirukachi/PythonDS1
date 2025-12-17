"""
Part4_Evaluation.py
-------------------------------------------------------------------------------
Part 4: Final Evaluation and Comparison

PROBLEM STATEMENT:
We have three contenders:
1. Overfit Model (No constraints).
2. Pre-Pruned Model (Grid Search).
3. Post-Pruned Model (Optimal Alpha).

STEPS TO SOLVE:
1. Evaluate all 3 on the held-out TEST set.
2. Build a Comparison Table showing Train/Val/Test Accuracy and Tree Complexity.
3. Conclude which model is best for production.

CONCEPTS:
- Test Set: Data never seen during training OR tuning. The final "Exam".
- Bias-Variance Tradeoff: Balancing Simplicity (High Bias) vs Complexity (High Variance).
-------------------------------------------------------------------------------
"""

import pandas as pd
from utils import load_and_split_data
from Part1_Overfitting import run_part1_overfitting
from Part2_PrePruning import run_part2_prepruning
from Part3_PostPruning import run_part3_postpruning

def run_evaluation():
    print("\n=== Part 4: Final Comparison ===")
    
    # 1. Get Data
    _, _, X_test, _, _, y_test, _ = load_and_split_data()
    
    # 2. Get Trained Models
    model_overfit = run_part1_overfitting()
    model_pre = run_part2_prepruning()
    model_post = run_part3_postpruning()
    
    models = {
        'Overfit (Baseline)': model_overfit,
        'Pre-Pruned (Grid)': model_pre,
        'Post-Pruned (Alpha)': model_post
    }
    
    results = []
    
    for name, clf in models.items():
        # Calc Metrics
        # NOTE: We approximate Train/Val scores here or could pass them in. 
        # For simplicity, we just measure Test score and Complexity here.
        test_acc = clf.score(X_test, y_test)
        depth = clf.get_depth()
        leaves = clf.get_n_leaves()
        
        results.append({
            'Model': name,
            'Test Accuracy': test_acc,
            'Depth': depth,
            'Leaves': leaves,
            'Status': "BEST" if test_acc > 0.9 else "POOR"
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n--- Final Comparison Table ---")
    print(df_res)
    
    print("\nCONCLUSION:")
    print("1. Overfit model has high depth and excessive leaves.")
    print("2. Pruned models are simpler (fewer leaves) but often match or beat Test Accuracy.")
    print("3. Pre-Pruning is faster; Post-Pruning is more precise.")

if __name__ == "__main__":
    run_evaluation()
