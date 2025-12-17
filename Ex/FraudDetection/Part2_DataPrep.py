"""
Part2_DataPrep.py
-------------------------------------------------------------------------------
Part 2: Data Preparation Pipeline

PROBLEM STATEMENT:
Raw data is rarely ready for Machine Learning. We have two main issues:
1. Feature Scaling: 'Amount' (0-25k) is much larger than 'V1' (-2 to 2). This biases Distance-based algorithms like KNN.
2. Data Splitting: We need separate sets for training and testing to simulate real-world performance.

STEPS TO SOLVE:
1. Load Data.
2. Initialize StandardScaler.
3. Apply scaling to 'Amount' and 'Time' columns.
4. Perform Stratified Train-Test Split (80/20).
5. Perform Validation Split on the Train set.

EXPECTED OUTPUT:
- 3 sets of data vectors: (X_train, y_train), (X_val, y_val), (X_test, y_test).
- All sets will maintain the 0.17% fraud ratio.
- 'Amount' and 'Time' columns will be centered around 0.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd

# WHAT: Tool for splitting arrays into random subsets.
from sklearn.model_selection import train_test_split

# WHAT: Tool for standardizing features (z-score normalization).
from sklearn.preprocessing import StandardScaler

from utils import load_data

def prepare_data():
    """
    Main function to execute Data Preparation.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print("\n=== Part 2: Data Preparation ===")
    
    # 1. Load Data
    df = load_data()
    if df is None: return None
    
    # WHAT: Separate inputs (Features) from output (Target).
    # drop('Class', axis=1) removes the target column.
    X = df.drop('Class', axis=1) 
    y = df['Class']              
    
    # ---------------------------------------------------------
    # Step 1: Feature Scaling
    # ---------------------------------------------------------
    print("[Step 1] Scaling 'Amount' and 'Time' features...")
    
    # WHAT: Initialize the Scaler.
    # WHY: StandardScaler standardizes features by removing the mean and scaling to unit variance.
    #      Formula: z = (x - mean) / std_dev
    scaler = StandardScaler()
    
    # WHAT: Apply scaling ONLY to 'Amount' and 'Time'.
    # WHY: Columns V1-V28 are already scaled (result of PCA transformation). Re-scaling them is unnecessary.
    # METHOD: fit_transform()
    #   - fit: Calculates mean and std deviation of the column.
    #   - transform: Applies the math.
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    
    # ---------------------------------------------------------
    # Step 2: Train-Test Split
    # ---------------------------------------------------------
    print("[Step 2] Splitting Data (80% Train, 20% Test)...")
    
    # WHAT: Divide data into Training and Test sets.
    # ARGUMENTS:
    # - test_size=0.2: 20% of data (approx 57k rows) goes to Test.
    # - stratify=y: CRITICAL for Imbalanced Data.
    #      It forces the split to preserve the % of Fraud in both sets.
    #      Without this, we might accidentally put ALL frauds in Train and 0 in Test.
    # - random_state=42: Fixes the random seed so results are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # ---------------------------------------------------------
    # Step 3: Validation Split
    # ---------------------------------------------------------
    print("[Step 3] Creating Validation Set from Training Data...")
    
    # WHAT: Further split the Training data to create a Validation set.
    # WHY: We use 'Train' to fit model, 'Val' to tune hyperparameters, and 'Test' ONLY for final report.
    # ARGUMENTS: Same as above.
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    # EXPECTED OUTPUT: Shapes of the resulting arrays.
    print(f"Final Shapes -> Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Return all sets + scaler (needed for production pipeline later)
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test, scaler

if __name__ == "__main__":
    prepare_data()
