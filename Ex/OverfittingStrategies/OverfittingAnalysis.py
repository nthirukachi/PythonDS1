"""
Problem Statement:
Using the housing prices dataset, Discuss strategies like cross-validation to avoid overfitting your regression model when working with the housing prices dataset.

Steps to Solve the Problem:
1.  Import necessary libraries: pandas, sklearn (model, selection).
2.  Load the dataset ('../Housing.csv').
3.  Preprocess the data (One-Hot Encoding).
4.  STRATEGY 1: Single Train-Test Split.
    -   Split data once (80/20).
    -   Train model.
    -   Compare Training Score vs Testing Score.
    -   Explanation: Large gap indicates overfitting.
5.  STRATEGY 2: K-Fold Cross-Validation.
    -   Split data into 5 folds.
    -   Calculate mean score.
    -   Explanation: Shows true model performance by averaging out randomness.
6.  Output the comparison and insights.

Sub-problems:
-   Data Handling.
-   Model Fitting.
-   Score Comparison.

Expected Output:
-   Train Score vs Test Score (Single Split).
-   Mean Cross-Validation Score.
-   Interpretation of whether the model is overfitting.
"""

# Importing libraries
# WHAT: Standard data science stack.
# WHY: pandas for data, sklearn for modeling.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

def analyze_overfitting_strategies():
    file_path = '../Housing.csv'

    try:
        # Loading Data
        # WHAT: Reading the csv file.
        df = pd.read_csv(file_path)
        print("Data Loaded.")

        # Preprocessing
        # WHAT: Converting text columns to numbers.
        # WHY: Math requires numbers.
        df_numeric = pd.get_dummies(df, drop_first=True)
        
        # Separating Target(y) and Features(X)
        y = df_numeric['price']
        X = df_numeric.drop('price', axis=1)

        print("\n--- Strategy 1: Single Train-Test Split ---")
        # WHAT: Splitting data strictly once.
        # ARGUMENTS: test_size=0.2 (20% holdout).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Scoring
        # METHOD: score(X, y)
        # WHAT: Returns R-squared value.
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"Training Score: {train_score:.4f}")
        print(f"Testing Score:  {test_score:.4f}")
        
        # ANALYSIS
        gap = train_score - test_score
        print(f"Gap: {gap:.4f}")
        if gap > 0.1:
            print("-> Interpretation: Significant gap. Potential OVERFITTING. Model memorized training data.")
        else:
            print("-> Interpretation: Small gap. Model generalizes well (or is underfitting if scores are low).")


        print("\n--- Strategy 2: Cross-Validation (The Solution) ---")
        # WHAT: Using K-Fold to validate.
        # WHY: To avoid the 'lucky split' problem of Strategy 1.
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # METHOD: cross_val_score
        # WHAT: Runs the split-train-test loop 5 times.
        cv_scores = cross_val_score(model, X, y, cv=kf)

        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        
        print("\nComparison:")
        print("Single Split gave us one view (Test Score).")
        print("Cross-Validation gave us the average of 5 views.")
        print("Using CV ensures we don't accidentally think a model is great just because we picked an easy test set.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_overfitting_strategies()
