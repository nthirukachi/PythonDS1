"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR For Healthcare Imbalance Project

PROBLEM STATEMENT:
We need to execute the full study on Class Imbalance:
1. Demonstrate baseline failure.
2. Test Resampling (Data Level).
3. Test Reweighting (Algo Level).
4. Recommend solution.

STEPS:
Run Part 1, Part 2, Part 3, Part 4 sequentially.

EXPECTED OUTPUT:
- Full logs of the study.
- Generation of the 'PR_Curve_Threshold.png' plot.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Import runner functions from all parts.
from Part1_Baseline import run_part1_baseline
from Part2_DataLevel import run_part2_datalevel
from Part3_AlgoLevel import run_part3_algolevel
from Part4_Evaluation import run_final_evaluation

def main():
    """
    Main entry point to run the Imbalance Study.
    """
    print("STARTING IMBALANCE PROJECT")
    print("==========================")
    
    # Step 1: Baseline
    run_part1_baseline()
    
    # Step 2: Data Resampling (SMOTE)
    run_part2_datalevel()
    
    # Step 3: Algorithm Tuning (Weights + Thresholds)
    run_part3_algolevel()
    
    # Step 4: Final Summary
    run_final_evaluation()
    
    print("\n==========================")
    print("PROJECT COMPLETED.")
    print("Plots Generated: ")
    print(" - PR_Curve_Threshold.png (Precision Recall Curve)")

if __name__ == "__main__":
    # WHAT: Execute only if run directly.
    main()
