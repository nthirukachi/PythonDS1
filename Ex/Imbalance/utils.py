"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT (Class Imbalance):
In healthcare, "Positive" cases (Disease, Readmission) are usually rare.
If we have 10,000 patients, maybe only 1,500 are sick.
This **85 vs 15** ratio is called "Imbalance".
Models love to cheat by saying "Everyone is Healthy" (85% Accuracy).
We need to generate a dataset that specifically mimics this problem to study solutions.

STEPS TO SOLVE:
1. Generate synthetic data using `make_classification`.
2. Force the class weights to be [0.85, 0.15].
3. Split into Train/Test, preserving this ratio (`stratify`).

EXPECTED OUTPUT:
- X_train, y_train with exactly ~15% Class 1.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# WHAT: Tool to create fake datasets.
from sklearn.datasets import make_classification

# WHAT: Tool to split data.
from sklearn.model_selection import train_test_split

def generate_imbalanced_data():
    """
    Generates a synthetic dataset with 85% Class 0 and 15% Class 1.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Generating Imbalanced Healthcare Data (Synthetic)...")
    
    # WHAT: Create synthetic matrix.
    # ARGUMENTS:
    # - n_samples=10000: Total patients.
    # - n_features=20: 20 vital signs/lab results.
    # - n_informative=15: 15 features actually signal the disease (Signal).
    # - weights=[0.85, 0.15]: ***CRITICAL***. Forces the 85% / 15% ratio.
    # - flip_y=0: Don't flip labels randomly. Keep the classes clean.
    # - random_state=42: Consistent result.
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        weights=[0.85, 0.15],
        random_state=42,
        flip_y=0 
    )
    
    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    # WHAT: Count unique values in y to prove it worked.
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Class Distribution: {dist} (Expect ~8500 vs ~1500)")
    
    # ---------------------------------------------------------
    # Splitting
    # ---------------------------------------------------------
    # WHAT: Split Training (80%) and Test (20%).
    # ARGUMENT: stratify=y.
    # WHY: We MUST keep the 15% ratio in the Test set. 
    # If we don't, the Test set might accidentally get 0 sick people.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Return the split data.
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    generate_imbalanced_data()
