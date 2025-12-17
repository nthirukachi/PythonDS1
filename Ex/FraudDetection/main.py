"""
main.py
-------------------------------------------------------------------------------
ORCHESTRATOR SCRIPT

PROBLEM STATEMENT:
Managing 7 different scripts individually is tedious. 
We need a single entry point to run the entire project lifecycle.

STEPS TO SOLVE:
1. Import entry functions from all Parts.
2. Execute proper sequence: EDA -> Training -> Tuning -> Analysis -> Deploy.

EXPECTED OUTPUT:
- Console logs from every part.
- Plots appearing sequentially.
- Final "PROJECT COMPLETED" message.
-------------------------------------------------------------------------------
"""

# WHAT: Import main functions from each module.
from Part1_EDA import run_eda
from Part3_ModelComparison import run_comparison
from Part4_Tuning import run_tuning
from Part5_Imbalance import run_imbalance_handling
from Part6_FeatureImportance import run_feature_importance
from Part7_Production import train_and_save_production_model

def main():
    print("STARTING CAPSTONE PROJECT: FRAUD DETECTION SYSTEM")
    print("=================================================")
    
    # 1. Exploratory Data Analysis
    # WHAT: Plots pie charts vs histograms.
    run_eda()
    
    # 2. Model Comparison
    # WHAT: Trains KNN, SVM, DT, RF and compares metrics.
    # INCLUDES: Part 2 (Data Prep) implicitly.
    run_comparison()
    
    # 3. Tuning
    # WHAT: GridSearch for best RF Recall.
    run_tuning()
    
    # 4. Imbalance Handling
    # WHAT: Test SMOTE/Thresholds.
    run_imbalance_handling()
    
    # 5. Interpretability
    # WHAT: Feature Importance & SHAP.
    run_feature_importance()
    
    # 6. Production
    # WHAT: Save model.pkl and test API.
    train_and_save_production_model()
    
    print("\nPROJECT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
