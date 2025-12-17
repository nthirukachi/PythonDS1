"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR SCRIPT

PROBLEM STATEMENT:
Managing 4 separate scripts manually (Part 1, 2, 3, 4) is tedious.
This script acts as the "Master Controller" to run the entire project in sequence.

STEPS TO SOLVE:
1. Import the final evaluation function (`run_evaluation`) from Part 4.
2. Calculate and display the total execution flow.

EXPECTED OUTPUT:
- It will trigger Part 1 (Overfitting Demo).
- It will trigger Part 2 (Pre-Pruning).
- It will trigger Part 3 (Post-Pruning).
- It will finally run Part 4 (Comparison) and print the Summary Table.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Import the function `run_evaluation` from Part4_Evaluation.py.
# WHY: Part 4 internally calls Parts 1, 2, and 3, so running Part 4 runs everything.
from Part4_Evaluation import run_evaluation

def main():
    """
    Main entry point for the Tree Pruning Project.
    """
    print("STARTING TREE PRUNING PROJECT")
    print("=============================")
    print("Sequence: Overfitting -> PrePruning -> PostPruning -> Comparison")
    
    # WHAT: Execute the evaluation pipeline defined in Part 4.
    # This chain-reacts to run all previous parts.
    run_evaluation()
    
    print("\n=============================")
    print("PROJECT COMPLETED SUCCESSFULLY.")
    print("Please check the folder for generated PNG plots:")
    print(" - Overfitting_Curve_Depth.png")
    print(" - PrePruning_Heatmap.png")
    print(" - PostPruning_AlphaCurve.png")

if __name__ == "__main__":
    # WHAT: Execute main() only if this file is run directly (not imported).
    main()
