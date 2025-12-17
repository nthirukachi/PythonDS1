"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT:
To build a machine learning model for loan defaults, we need a dataset.
However, raw real-world data is rarely ready for algorithms. It often contains:
1.  String values (e.g., "good", "bad", "high").
2.  Missing values (NaNs).
3.  Inconsistent formatting.

We need a "Data Pipeline" to:
1.  Fetch the "German Credit" dataset (a standard benchmark for credit risk).
2.  Encode target labels ("bad" -> 1, "good" -> 0) so the computer understands Risk.
3.  Encode feature columns (Strings -> Numbers).
4.  Split the data into Training and Testing sets to ensure we don't cheat.

STEPS TO SOLVE:
1.  Import `fetch_openml` to download data directly from the internet.
2.  Import `LabelEncoder` and `OrdinalEncoder` to transform text to numbers.
3.  Load the dataset and identify Categorical vs Numerical columns.
4.  Apply transformations.
5.  Return clean arrays (X_train, X_test, y_train, y_test).

EXPECTED OUTPUT:
- X_train, X_test: Pandas DataFrames containing only numbers (integers/floats).
- y_train, y_test: Numpy arrays containing 0s and 1s.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Import `ssl` module.
# WHY: Python's default security settings often block downloads from OpenML on corporate networks.
# WHEN: Used whenever you get "SSL: CERTIFICATE_VERIFY_FAILED" errors.
import ssl

# WHAT: Import Pandas.
# WHY: To handle tabular data (Rows/Columns) efficiently.
import pandas as pd

# WHAT: Import dataset fetcher.
# ARGUMENTS: 'credit-g' is the ID for German Credit data.
from sklearn.datasets import fetch_openml

# WHAT: Import Encoders.
# LabelEncoder: Converts Target (y) ["bad", "good"] -> [1, 0].
# OrdinalEncoder: Converts Features (X) ["low", "high"] -> [0, 1].
# WHY: Mathematical models (like Random Forest) require numerical input.
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# WHAT: Import Splitter.
# WHY: We must hide some data (Test Set) to evaluate the model fairly later.
from sklearn.model_selection import train_test_split

# FIX: Disable SSL verification globally for this script.
# WHY: Prevents script failure due to firewall/certificate issues.
ssl._create_default_https_context = ssl._create_unverified_context


def load_loan_data():
    """
    Fetches and preprocesses the German Credit dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("Loading German Credit Data (Loan Default Proxy)...")
    
    # ---------------------------------------------------------
    # 1. Fetch Data
    # ---------------------------------------------------------
    # WHAT: Download the dataset.
    # ARGUMENTS:
    # - name='credit-g': The specific dataset name on OpenML.
    # - version=1: Ensures we get the standard version, not a variant.
    # - as_frame=True: Returns a Pandas DataFrame instead of a raw dictionary.
    # EXPECTED OUTPUT: A 'Bunch' object containing .data (X) and .target (y).
    dataset = fetch_openml('credit-g', version=1, as_frame=True)
    
    X = dataset.data
    y = dataset.target
    feature_names = X.columns.tolist()
    
    print(f"Raw Data Shape: {X.shape}")
    
    # ---------------------------------------------------------
    # 2. Encode Target (y)
    # ---------------------------------------------------------
    # WHAT: Initialize Label Encoder.
    # WHY: The target 'class' has values 'good' and 'bad'. Models need 0 and 1.
    le = LabelEncoder()
    
    # WHAT: Transform strings to numbers.
    # EXPECTED OUTPUT: Array of integers [0, 1, 0, 0, 1...].
    y_encoded = le.fit_transform(y)
    
    # WHAT: Check mapping.
    # WHY: We want 'bad' (Risk) to be Class 1 (Positive Class).
    # If standard sorting assigned 'bad' -> 0, we must invert it.
    # le.classes_ stores the unique labels in alphabetical order: ['bad', 'good'].
    # So 'bad' is usually 0. We want 'bad' to be 1 (The thing we are detecting).
    if le.classes_[0] == 'bad':
        # Currently bad=0. We invert (1 - 0 = 1).
        y_encoded = 1 - y_encoded
        print("Encoded Target: 'bad' mapped to 1 (Default), 'good' to 0.")
    
    # ---------------------------------------------------------
    # 3. Encode Features (X)
    # ---------------------------------------------------------
    # WHAT: Identify text columns.
    # METHOD: .select_dtypes(include=['object', 'category']) finds Non-Numeric cols.
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # WHAT: Initialize Ordinal Encoder.
    # WHY: Trees handle ordinal numbers (0, 1, 2) well. One-Hot encoding is also an option
    # but Ordinal is compact and standard for tree-based models (like Random Forest).
    oe = OrdinalEncoder()
    
    # WHAT: Transform categorical columns.
    X[cat_cols] = oe.fit_transform(X[cat_cols])
    
    print("Categorical variables encoded successfully.")
    
    # ---------------------------------------------------------
    # 4. Split Data
    # ---------------------------------------------------------
    # WHAT: Split data into Training (80%) and Testing (20%).
    # ARGUMENTS:
    # - test_size=0.2: 20% of data is held out for the Final Exam.
    # - stratify=y_encoded: Ensures the % of Defaults is consistent in Train and Test.
    #   If 30% of people default in total, 30% of Train and 30% of Test will default.
    # - random_state=42: Reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    print(f"Train Size: {len(X_train)} samples")
    print(f"Test Size:  {len(X_test)} samples")
    
    # RETURN: The prepared data blocks ready for machine learning.
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    load_loan_data()
