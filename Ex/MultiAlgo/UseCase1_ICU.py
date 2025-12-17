"""
UseCase1_ICU.py
-------------------------------------------------------------------------------
USE CASE 1: ICU Readmission Prediction

PROBLEM STATEMENT:
We need to predict if a patient discharged from ICU will need to be readmitted.
This is a **Binary Classification** problem (Yes/No).
Constraint: "Interpretability". Physicians must trust the model. "Black box" models 
like Deep Learning are often rejected in favor of transparent ones.

STEPS TO SOLVE:
1. Load Synthetic Binary Data (from utils).
2. Train 3 Models:
   - Decision Tree (Highly Interpretable).
   - Random Forest (High Performance).
   - SVM (Good for high dimensions).
3. Evaluate using **Recall**. (We must minimize False Negatives - missing a sick patient).
4. Visualize the Decision Tree logic flow.
5. Extract Feature Importance to show doctors "What matters".

EXPECTED OUTPUT:
- A text report comparing Recall scores.
- A PNG image of the Decision Tree.
- A ranking of top clinical features.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# WHAT: Import Classifiers.
# Decision Tree: Builds a flowchart-like structure.
# Random Forest: Builds many trees and averages them.
# SVC: Finds a geometric boundary.
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# WHAT: Metrics.
# Recall: TP / (TP + FN). Crucial for safety.
from sklearn.metrics import classification_report, recall_score
from utils import generate_icu_data

def run_icu_use_case():
    print("\n=== USE CASE 1: ICU READMISSION PREDICTION ===")
    
    # ---------------------------------------------------------
    # 1. Data Loading & Splitting
    # ---------------------------------------------------------
    # WHAT: Generate 10k synthetic patient records.
    X, y = generate_icu_data()
    
    # WHAT: Split data into Training and Testing sets.
    # ARGUMENTS:
    # - test_size=0.2: 20% validation.
    # - stratify=y: KEEPS the 15% readmission rate constant in both sets.
    # - random_state=42: Reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # ---------------------------------------------------------
    # 2. Model Definition
    # ---------------------------------------------------------
    # WHAT: Dictionary storing our 3 candidate algorithms.
    models = {
        # Model 1: Decision Tree
        # ARG: max_depth=3. Important! Keeps tree small enough to be readable by a human.
        # ARG: class_weight='balanced'. Penalizes missing the rare 'Readmit' class.
        'Decision Tree': DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42),
        
        # Model 2: Random Forest
        # ARG: n_estimators=100. Uses 100 trees for stability.
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        
        # Model 3: SVM
        # ARG: kernel='linear'. Often easier to explain than RBF curvature.
        'SVM': SVC(kernel='linear', class_weight='balanced', random_state=42)
    }
    
    results = []
    
    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    for name, model in models.items():
        print(f"Training {name}...")
        
        # WHAT: Train model.
        model.fit(X_train, y_train)
        
        # WHAT: Predict on unseen test data.
        y_pred = model.predict(X_test)
        
        # WHAT: Calculate Metric.
        # WHY RECALL? If we predict "Healthy" but patient returns (FN), patient might die.
        # We optimize for catching the positive class.
        rec = recall_score(y_test, y_pred)
        
        results.append({'Algorithm': name, 'Recall': rec})
        print(f"-> {name} Recall: {rec:.4f}")

    print("\nMetrics Summary:")
    print(pd.DataFrame(results))

    # ---------------------------------------------------------
    # 4. Interpretability Task (Visualizing The Logic)
    # ---------------------------------------------------------
    print("\n[Analysis] Visualizing Decision Tree for Physician Review...")
    dt = models['Decision Tree']
    
    # WHAT: Configure Plot.
    plt.figure(figsize=(12, 6))
    
    # WHAT: Draw the tree structure.
    # ARGS: 
    # - filled=True: Colors nodes based on purity (Blue=Healthy, Orange=Readmit).
    # - feature_names: Labels for the flowchart nodes.
    plot_tree(dt, filled=True, feature_names=[f'Feat{i}' for i in range(20)], class_names=['No Readmit', 'Readmit'])
    plt.title("Decision Tree Flowchart (Physician Friendly)")
    
    # WHAT: Save to disk.
    plt.savefig('ICU_Decision_Tree.png')
    print("Decision Tree plot saved as ICU_Decision_Tree.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 5. Feature Importance (Risk Factors)
    # ---------------------------------------------------------
    print("\n[Analysis] Extracting Top Clinical Risk Factors...")
    rf = models['Random Forest']
    
    # WHAT: Extract importance scores (0 to 1).
    importances = rf.feature_importances_
    
    # WHAT: Create Series for sorting.
    feat_imp = pd.Series(importances, index=[f'Feat{i}' for i in range(20)]).sort_values(ascending=False)
    
    print("Top 5 Risk Factors (Drivers of Readmission):")
    print(feat_imp.head())
    
    print("\nCONCLUSION FOR USE CASE 1:")
    print("- Random Forest usually has higher Recall (Safety).")
    print("- Decision Tree is best for EXPLAINING the connection to a doctor.")
    print("- Recommendation: Use RF for backend screening, DT for explanation interface.")

if __name__ == "__main__":
    run_icu_use_case()
