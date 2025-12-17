"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT:
To experiment with Machine Learning, we first need to:
1. Load a dataset.
2. Split it into Training, Validation, and Test sets.
This file serves as a shared "Data Utility" module for all other scripts.

STEPS TO SOLVE:
1. Import `load_breast_cancer` from sklearn.
2. Import `train_test_split` for dividing data.
3. Define `load_and_split_data()` to perform a 70% Train, 15% Val, 15% Test split.
4. Define `plot_learning_curve()` to visualize model performance.

CONCEPTS & OUTPUT:
- Stratification: Ensuring each split has the same proportion of Cancer/No-Cancer cases.
- Data Leakage: Why we strictly separate Test data.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Import pandas for table manipulations.
# WHY: Standard tool for Data Science.
import pandas as pd

# WHAT: Import numpy for numerical operations (arrays).
# WHY: Scikit-learn uses numpy arrays under the hood.
import numpy as np

# WHAT: Import matplotlib for plotting.
# WHY: We need to visualize Learning Curves.
import matplotlib.pyplot as plt

# WHAT: Import standard Breast Cancer dataset.
# WHY: It is a trusted, built-in binary classification dataset (Malignant vs Benign).
from sklearn.datasets import load_breast_cancer

# WHAT: Import splitting tool.
# WHY: To randomly divide rows into subsets.
from sklearn.model_selection import train_test_split

def load_and_split_data():
    """
    Loads data and creates a 3-way split (Train/Val/Test).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    print("Loading Breast Cancer Data...")
    
    # WHAT: Load the raw data object.
    # ARGUMENT: return_X_y=False (default) returns a Bunch object with metadata.
    data = load_breast_cancer()
    
    # WHAT: Extract Features (X) and Target (y).
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # EXPECTED OUTPUT: Tuple e.g. (569, 30)
    print(f"Dataset Shape: {X.shape}")
    
    # ---------------------------------------------------------
    # Split 1: Isolate Training Data (70%)
    # ---------------------------------------------------------
    # WHAT: Split total data into "Train" and "Temp".
    # ARGUMENTS:
    # - test_size=0.3: 30% goes to Temp (to be used for Val/Test), 70% stays in Train.
    # - stratify=y: Crucial. Ensures Malignant/Benign ratio is same in Train and Temp.
    # - random_state=42: Ensures we get the exact same split every time we run.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # ---------------------------------------------------------
    # Split 2: Divide Temp into Val (15%) and Test (15%)
    # ---------------------------------------------------------
    # WHAT: Split the 30% "Temp" chunk into two equal halves.
    # LOGIC: 0.3 * 0.5 = 0.15. So X_val is 15% of total, X_test is 15% of total.
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # WHAT: Print sizes for verification.
    print(f"Train samples: {len(X_train)} (70%)")
    print(f"Val samples:   {len(X_val)} (15%)")
    print(f"Test samples:  {len(X_test)} (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def plot_learning_curve(depths, train_scores, val_scores, title="Learning Curve"):
    """
    Helper function to draw line charts of accuracy vs depth.
    
    ARGUMENTS:
    - depths: List of x-axis values (e.g., [1, 2, 3...]).
    - train_scores: List of y-axis values for Training.
    - val_scores: List of y-axis values for Validation.
    """
    # WHAT: Create a figure window.
    plt.figure(figsize=(10, 6))
    
    # WHAT: Plot Training Accuracy Line.
    # ARG: marker='o' puts dots on data points.
    plt.plot(depths, train_scores, marker='o', label='Train Accuracy')
    
    # WHAT: Plot Validation Accuracy Line.
    # ARG: color='orange' to distinguish from Train.
    plt.plot(depths, val_scores, marker='o', label='Validation Accuracy', color='orange')
    
    # WHAT: Add labels and title for readability.
    plt.xlabel('Tree Depth / Complexity')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend() # Shows the small box explaining which color is which.
    plt.grid(True) # Adds gridlines for easier reading.
    
    # WHAT: Save plot to disk.
    # WHY: Allows user to view it later without blocking script execution.
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close() # Free up memory.

if __name__ == "__main__":
    # WHAT: Test run to verify splitting logic works.
    load_and_split_data()
