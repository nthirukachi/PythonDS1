"""
Part1_EDA.py
-------------------------------------------------------------------------------
Part 1: Exploratory Data Analysis (EDA)

PROBLEM STATEMENT:
We need to understand the characteristics of the Credit Card Fraud dataset before
training any models. Specifically, we need to confirm the severity of the Class
Imbalance (Fraud vs Legit) and check data distributions.

STEPS TO SOLVE:
1. Load the dataset using `utils.py`.
2. Analyze Class Imbalance (Step 2.1).
3. Visualize Transaction Amounts (Step 2.2).
4. Analyze Feature Correlation (Step 2.3).

SUB-PROBLEMS:
- How to visualize a 99.8% vs 0.2% ratio? (Pie Chart)
- How to visualize amounts ranging from $0 to $25,000? (Log Scale)

EXPECTED OUTPUT:
- Statistical summary of the Fraud Percentage.
- Pie Chart showing the imbalance.
- Histogram showing Amount distribution.
- Heatmap showing feature correlations.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
# WHAT: Standard visualization libraries.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# WHAT: Importing our custom loader.
# WHY: To reuse the safe loading logic defined in `utils.py`.
from utils import load_data

def run_eda():
    """
    Main function to execute Exploratory Data Analysis.
    """
    print("\n=== Part 1: Exploratory Data Analysis ===")
    
    # WHAT: Load the dataframe.
    # EXPECTED OUTPUT: DataFrame object `df`.
    df = load_data()
    # WHAT: Check for load failure.
    if df is None: return
    
    # ---------------------------------------------------------
    # Step 1: Analyze Class Distribution (Imbalance)
    # ---------------------------------------------------------
    print("\n[Step 1] Analyzing Class Distribution...")
    
    # WHAT: Count occurrences of each class.
    # METHOD: .value_counts() on a Series.
    # WHY: To see raw numbers of Legit (0) vs Fraud (1).
    # EXPECTED OUTPUT: Series like {0: 284315, 1: 492}
    counts = df['Class'].value_counts()
    print(counts)
    
    # WHAT: Calculate percentage.
    # WHY: Raw numbers are hard to grasp; % is clearer.
    # CALCULATION: (Fraud Count / Total Rows) * 100
    fraud_pct = counts[1] / len(df) * 100
    print(f"Fraud Percentage: {fraud_pct:.4f}%")
    print("Observation: This is an EXTREMELY imbalanced dataset.")
    
    # VISUALIZATION: Pie Chart
    # WHAT: Create a circular chart divided into sectors.
    # ARGUMENTS:
    # - x: `counts` (the data).
    # - labels: Names for the slices.
    # - autopct='%1.2f%%': Format the label to show 2 decimal places.
    # - startangle=90: Rotates chart so the split is visible at the top.
    # - colors: Custom hex codes for blue and red.
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['Legitimate (0)', 'Fraud (1)'], autopct='%1.2f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Class Imbalance: Legitimate vs Fraud')
    plt.show()
    
    # ---------------------------------------------------------
    # Step 2: Transaction Amount Distribution
    # ---------------------------------------------------------
    print("\n[Step 2] Analyzing Transaction Amounts...")
    
    # WHAT: Create a Histogram.
    # METHOD: sns.histplot().
    # ARGUMENTS:
    # - data: `df` (The dataframe).
    # - x: 'Amount' (column to plot).
    # - hue: 'Class' (colors the bars differently for Fraud vs Legit).
    # - bins=50: Divide the range into 50 bars.
    # - log_scale=(False, True): IMPORTANT. It sets the Y-axis (Frequency) to Log Scale.
    # WHY LOG SCALE? 
    #   Legit transactions are in the millions. Fraud transactions are in the hundreds.
    #   On a linear scale, the Fraud bars would be 1 pixel tall (invisible).
    #   Log scale compresses the big numbers so we can compare the shapes.
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Amount', hue='Class', bins=50, log_scale=(False, True))
    plt.title('Transaction Amount Distribution (Log Scale Frequency)')
    plt.xlabel('Amount ($)')
    plt.ylabel('Frequency (Log Scale)')
    plt.show()
    
    # ---------------------------------------------------------
    # Step 3: Correlation Analysis
    # ---------------------------------------------------------
    print("\n[Step 3] Calculating Feature Correlations...")
    
    # WHAT: Calculate Pearson Correlation Coefficient between all columns.
    # METHOD: df.corr()
    # WHY: To see if Feature A increases when Feature B increases (Redundancy) 
    #      or if Feature A correlates with Class (Predictive Power).
    # EXPECTED OUTPUT: A 31x31 matrix of numbers between -1 and 1.
    corr = df.corr()
    
    # VISUALIZATION: Heatmap
    # WHAT: Visualizing the matrix as colors.
    # ARGUMENTS:
    # - cmap='coolwarm': Red for +1, Blue for -1.
    # - vmin=-1, vmax=1: Set the anchor points for the colors.
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    print("\nKey Findings from EDA:")
    print("1. Imbalance is severe (0.17%). Accuracy metric will be meaningless.")
    print("2. 'V' features are PCA components (uncorrelated with each other).")
    print("3. 'Amount' and 'Time' have different scales than 'V' features -> Need Scaling.")

if __name__ == "__main__":
    run_eda()
