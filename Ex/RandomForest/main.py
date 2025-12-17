"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR For Random Forest Project

PROBLEM STATEMENT:
We have multiple modular scripts (Comparison, Interpretation, Tuning).
We need a single entry point to run them all in logical order to generate
the full analysis report.

STEPS:
1. Run Part 2 (because Part 2 imports Part 1, this runs both).
2. Run Part 3 (Tuning).
3. Confirm completion and list generated artifacts (Plots).

EXPECTED OUTPUT:
- Sequential execution logs in console.
- Creation of PNG plots in the directory.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Import main functions from our modules.
from Part2_Interpretation import run_part2_interpretation
from Part3_Tuning import run_part3_tuning

def main():
    """
    Main execution function.
    """
    print("STARTING RANDOM FOREST PROJECT")
    print("==============================")
    
    # STEP 1: Run Comparison & Interpretation
    # Note: Part 2 calls `run_part1_comparison()` internally to get the model.
    # So this one line executes Part 1 AND Part 2.
    run_part2_interpretation()
    
    # STEP 2: Run Hyperparameter Tuning
    run_part3_tuning()
    
    print("\n==============================")
    print("PROJECT COMPLETED.")
    print("Plots Generated:")
    print(" - RF_Feature_Importance.png (Global Importance)")
    print(" - SHAP_Summary.png (Local Importance - Check if created)")

if __name__ == "__main__":
    # WHAT: Execute only if run directly.
    main()
