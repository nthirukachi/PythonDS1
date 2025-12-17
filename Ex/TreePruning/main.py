"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR

PROBLEM: Run the full Tree Pruning study.
STEPS: Call Part 4 (which internally calls 1, 2, and 3).
-------------------------------------------------------------------------------
"""

from Part4_Evaluation import run_evaluation

def main():
    print("STARTING TREE PRUNING PROJECT")
    run_evaluation()
    print("PROJECT COMPLETED.")

if __name__ == "__main__":
    main()
