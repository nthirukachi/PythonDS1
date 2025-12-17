"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT:
To analyze Overfitting, we need a clean dataset split into three parts:
1. Training Set: To teach the model (70%).
2. Validation Set: To tune hyperparameters (e.g., pruning alphas) (15%).
3. Test Set: To evaluate final performance (15%).

STEPS TO SOLVE:
1. Load Breast Cancer dataset from Scikit-Learn.
2. Perform a first split (Train vs Temp) 70/30.
3. Perform a second split on Temp (Val vs Test) 50/50.

CONCEPTS & OUTPUT:
- Data Splitting: Essential to prevent "Data Leakage" and ensure the model generalizes.
- Output: X_train, X_val, X_test, y_train, y_val, y_test.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Import standard libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_split_data():
    """
    Loads Breast Cancer data and splits it 70/15/15.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("Loading Breast Cancer Data...")
    
    # WHAT: Load dataset.
    # ARGS: return_X_y=True gives us two arrays (Features, Labels) directly.
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Dataset Shape: {X.shape}")
    
    # ---------------------------------------------------------
    # Step 1: Split into Train (70%) and Remaining (30%)
    # ---------------------------------------------------------
    # ARGUMENTS:
    # - test_size=0.3: Reserve 30% for Val+Test.
    # - random_state=42: Reproducibility.
    # - stratify=y: Maintain the ratio of Malignant/Benign samples.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # ---------------------------------------------------------
    # Step 2: Split Remaining (30%) into Val (15%) and Test (15%)
    # ---------------------------------------------------------
    # ARGUMENTS:
    # - test_size=0.5: Split the 30% chunk exactly in half.
    # 0.3 * 0.5 = 0.15 (15% of total).
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"Train samples: {len(X_train)} (70%)")
    print(f"Val samples:   {len(X_val)} (15%)")
    print(f"Test samples:  {len(X_test)} (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def plot_learning_curve(depths, train_scores, val_scores, title="Learning Curve"):
    """
    Helper to plot Training vs Validation Accuracy over Tree Depth.
    
    WHAT: Visualizes Overfitting.
    
    EXPECTED OUTPUT:
    - Train score line goes to 1.0 (Perfect memorization).
    - Val score line goes up then drops (Overfitting).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, marker='o', label='Train Accuracy')
    plt.plot(depths, val_scores, marker='o', label='Validation Accuracy', color='orange')
    plt.xlabel('Tree Depth / Complexity')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save instead of show
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close()

if __name__ == "__main__":
    load_and_split_data()
