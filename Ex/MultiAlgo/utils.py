"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT:
In a Multi-Algorithm study, we often need consistent data sources and visualization
tools across different experiments.
Since we lack external CSV files for this project, we must generate synthetic data
that mimics the statistical properties of real clinical datasets.

STEPS TO SOLVE:
1. Import data generation libraries (sklearn).
2. Implement `generate_icu_data` for Binary Classification (Imbalanced).
3. Implement `generate_triage_data` for Multi-Class Classification.
4. Implement `generate_health_data` for Regression/Imputation tasks.
5. Create a shared `plot_confusion_matrix` tool.

CONCEPTS & OUTPUT:
- Synthetic Data: mathematically generated rows/columns used for testing logic.
- Imbalance: when one class (ICU Readmission) is much rarer than another.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Import Numpy.
# WHY: Fundamental package for scientific computing and array manipulation.
import numpy as np

# WHAT: Import Pandas.
# WHY: Used for DataFrames (table-like structures) to hold our generated data.
import pandas as pd

# WHAT: Import Matplotlib.
# WHY: Core plotting library for Python.
import matplotlib.pyplot as plt

# WHAT: Import Seaborn.
# WHY: Wrapper for Matplotlib that makes heatmaps (confusion matrices) easier to read.
import seaborn as sns

# WHAT: Import generation tools.
# WHY: `make_classification` creates fake classification problems.
#      `load_diabetes` gives us a standard regression dataset.
from sklearn.datasets import make_classification, load_diabetes

# ----------------- FUNCTIONS -----------------

def generate_icu_data():
    """
    Generates synthetic data for Use Case 1: ICU Readmission.
    
    WHAT: Creates a binary classification dataset (0=No Readmit, 1=Readmit).
    
    WHY: To simulate a clinical scenario where readmissions are rare (15%) 
         but critical to detect.
         
    RETURNS:
        X (ndarray): Feature matrix (10,000 patients x 20 clinical features).
        y (ndarray): Target vector (0 or 1).
    """
    print("Generating Synthetic ICU Data (Binary)...")
    
    # WHAT: Usage of `make_classification`
    # ARGUMENTS:
    # - n_samples=10000: Large n to ensure statistical significance.
    # - n_features=20: Simulating 20 clinical vitals (BP, HR, etc).
    # - n_informative=15: Only 15 features actually predict the outcome (Signal).
    # - weights=[0.85, 0.15]: FORCE Imbalance. 85% Class 0, 15% Class 1.
    # - random_state=42: Ensures we get the exact same "random" data every time.
    X, y = make_classification(
        n_samples=10000, 
        n_features=20,       
        n_informative=15,    
        weights=[0.85, 0.15], 
        random_state=42
    )
    
    # EXPECTED OUTPUT: X shape (10000, 20), y shape (10000,)
    return X, y

def generate_triage_data():
    """
    Generates data for Use Case 2: ED Triage.
    
    WHAT: Creates a Multi-Class classification dataset (5 Classes).
    
    WHY: Simulates an Emergency Dept where patients are Triage Level 1 to 5.
    
    RETURNS:
        X, y: Features and Target labels (0, 1, 2, 3, 4).
    """
    print("Generating Synthetic Triage Data (5-Class)...")
    
    # WHAT: Generate 5 distinct classes.
    # ARGUMENTS:
    # - n_classes=5: Target y will have values 0,1,2,3,4.
    # - weights=[...]: Simulates real ED distribution (e.g. L3/L4 are most common).
    # - n_clusters_per_class=1: Keeps data relatively simple (linear separability).
    X, y = make_classification(
        n_samples=10000, 
        n_features=20,
        n_informative=15,
        n_classes=5,
        n_clusters_per_class=1,
        weights=[0.05, 0.15, 0.40, 0.30, 0.10], 
        random_state=42
    )
    return X, y

def generate_health_data():
    """
    Generates data for Use Case 3: Population Health.
    
    WHAT: Loads standard Diabetes data and INJECTS Missing Values.
    
    WHY: Real healthcare data is messy. We need to test if algorithms can handle NaNs.
    
    RETURNS:
        X_df (DataFrame): Features with ~10% NaN values.
        y (ndarray): Target variable (Disease progression).
    """
    print("Generating Health Data with Missing Values...")
    
    # WHAT: Load reliable external dataset from sklearn library.
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # WHAT: Convert to DataFrame for easier manipulation.
    # WHY: DataFrames allow named columns, unlike numpy arrays.
    feature_names = data.feature_names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # ---------------------------------------------------------
    # SUB-PROBLEM solution: How to create simulated errors (NaNs)?
    # ---------------------------------------------------------
    
    # WHAT: Create a Random Number Generator.
    rng = np.random.RandomState(42)
    
    # WHAT: Calculate how many cells should be empty.
    # Calc: TotalCells * 0.1 (10%).
    n_missing = int(X.size * 0.1) 
    
    # WHAT: Pick random Row and Col indices.
    row_idxs = rng.randint(0, X.shape[0], n_missing)
    col_idxs = rng.randint(0, X.shape[1], n_missing)
    
    # WHAT: Insert np.nan (Not-a-Number) at those coordinates.
    X_df.values[row_idxs, col_idxs] = np.nan
    
    return X_df, y

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Helper to plot a confusion matrix.
    
    ARGS:
    - cm: The matrix (array).
    - labels: List of strings (e.g. ['Urgent', 'Non-Urgent']).
    """
    plt.figure(figsize=(8, 6))
    
    # WHAT: Seaborn Heatmap.
    # ARGS: annot=True (Show numbers), fmt='d' (Integers).
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # WHAT: Save file instead of blocking execution.
    # WHY: plt.show() pauses the script. In automation, we prefer saving files.
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved successfully as: {filename}")
    plt.close()

if __name__ == "__main__":
    # Test generation
    generate_icu_data()
    generate_triage_data()
    generate_health_data()
