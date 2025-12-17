"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR SCRIPT

PROBLEM STATEMENT:
Managing 3 separate scripts manually is inefficient. 
We need a single entry point to run the entire Multi-Use-Case study.

STEPS TO SOLVE:
1. Import the `run` functions from the 3 modules.
2. Execute them sequentially.
3. Separate outputs with clear visual dividers.

EXPECTED OUTPUT:
- Console logs for ICU, Triage, and Health cases.
- Saved PNG plots in the directory.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Import the specific functions we defined in other files.
# WHY: Keeps this main script clean and readable.
from UseCase1_ICU import run_icu_use_case
from UseCase2_Triage import run_triage_use_case
from UseCase3_Health import run_health_use_case

def main():
    print("STARTING MULTI-ALGORITHM COMPARISON")
    print("-----------------------------------")
    
    # ---------------------------------------------------------
    # Use Case 1: Binary / ICU
    # ---------------------------------------------------------
    # Focus: Recall & Interpretability (Decision Tree).
    run_icu_use_case()
    
    print("\n" + "="*50 + "\n")
    
    # ---------------------------------------------------------
    # Use Case 2: Multi-Class / Triage
    # ---------------------------------------------------------
    # Focus: Latency & F1 Score.
    run_triage_use_case()
    
    print("\n" + "="*50 + "\n")
    
    # ---------------------------------------------------------
    # Use Case 3: Missing Data / Health
    # ---------------------------------------------------------
    # Focus: Imputation vs Native Handling.
    run_health_use_case()
    
    print("\nPROJECT COMPLETED.")
    print("Please check the folder for generated PNG plots.")

if __name__ == "__main__":
    main()
