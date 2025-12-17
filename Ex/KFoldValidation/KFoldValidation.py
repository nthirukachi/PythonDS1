"""
Problem Statement:
Using the housing prices dataset, Implement a Python function that performs k-fold cross-validation on your multiple regression model using the housing prices dataset provided.

Steps to Solve the Problem:
1.  Import necessary libraries: pandas (data), sklearn (model & validation).
2.  Load the dataset ('Housing.csv').
3.  Preprocess the data:
    -   Convert categorical variables to numeric (One-Hot Encoding).
    -   Separate features (X) and target variable (y - 'price').
4.  Initialize the Linear Regression model.
5.  Configure K-Fold Cross-Validation (typically k=5 or k=10).
6.  Perform Cross-Validation using `cross_val_score`.
    -   This splits the data into 'k' parts.
    -   Trains on k-1 parts, tests on 1 part.
    -   Repeats this 'k' times.
7.  Collect and print the scores (R-squared) for each fold.
8.  Calculate and print the mean score and standard deviation to assess model stability.

Sub-problems:
-   Data preparation.
-   K-Fold configuration.
-   Score aggregation.

Expected Output:
-   A list of R-squared scores for each of the k folds.
-   The average R-squared score.
-   The standard deviation of the scores (indicating how consistent the model is).
"""

# Importing pandas.
# WHAT: Data manipulation library.
# WHY: To load the CSV and prepare the dataframe.
# WHEN: Starting any data analysis project.
# EXPECTED OUTPUT: `pd` module available.
import pandas as pd

# Importing LinearRegression.
# WHAT: The predictive model algorithm.
# WHY: We are solving a regression problem (predicting continuous price).
# WHEN: The target variable is continuous.
# EXPECTED OUTPUT: `LinearRegression` class available.
from sklearn.linear_model import LinearRegression

# Importing KFold.
# WHAT: Provides train/test indices to split data in train/test sets.
# WHY: To define how we want to split the data (how many folds, whether to shuffle).
# WHEN: Setting up the cross-validation strategy.
# EXPECTED OUTPUT: `KFold` class available.
from sklearn.model_selection import KFold

# Importing cross_val_score.
# WHAT: Function to evaluate a score by cross-validation.
# WHY: To automate the loop of splitting, training, testing, and scoring.
# WHEN: You want a robust estimate of model performance.
# EXPECTED OUTPUT: `cross_val_score` function available.
from sklearn.model_selection import cross_val_score

# Importing numpy.
# WHAT: Numerical computing library.
# WHY: To calculate mean and standard deviation of the scores easily.
# WHEN: Performing aggregate math on arrays.
# EXPECTED OUTPUT: `np` module available.
import numpy as np

def perform_kfold_validation():
    # Defining file path.
    # WHAT: Path to the dataset.
    # EXPECTED OUTPUT: String 'Housing.csv'.
    file_path = 'Housing.csv'

    try:
        # Loading data.
        # METHOD: pd.read_csv(file_path)
        # ARGUMENTS:
        #   - file_path: 'Housing.csv'.
        #     WHY: The location of the data.
        # WHAT: Reads the file into a DataFrame.
        # EXPECTED OUTPUT: DataFrame `df`.
        df = pd.read_csv(file_path)
        print("Dataset loaded.")

        # Preprocessing: convert categorical to numeric.
        # METHOD: pd.get_dummies(data, drop_first)
        # ARGUMENTS:
        #   - data: `df`.
        #   - drop_first: True.
        #     WHY: To avoid multicollinearity by removing one category per feature.
        #     WHEN: Using linear models like Linear Regression.
        # WHAT: Transforms text columns to binary 0/1 columns.
        # EXPECTED OUTPUT: Numeric DataFrame `df_numeric`.
        df_numeric = pd.get_dummies(df, drop_first=True)

        # Splitting Target and Features.
        # WHAT: Separating what we predict (price) from what we use to predict (area, bedrooms, etc).
        y = df_numeric['price']
        X = df_numeric.drop('price', axis=1)

        # Initializing the Model.
        # METHOD: LinearRegression()
        # WHAT: Creates the model object.
        # EXPECTED OUTPUT: `model` object ready to be fit.
        model = LinearRegression()

        # Configuring K-Fold.
        # METHOD: KFold(n_splits, shuffle, random_state)
        # ARGUMENTS:
        #   - n_splits: 5. 
        #     WHY: Standard balance between computation time and bias.
        #     WHEN: You have a moderate amount of data.
        #   - shuffle: True. 
        #     WHY: To mix the data before splitting. Vital if the data is sorted by date or price.
        #   - random_state: 42. 
        #     WHY: To ensure we get the same random splits every time we run this.
        # WHAT: Creates a cross-validation object.
        # EXPECTED OUTPUT: `kf` object.
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        print("\nStarting K-Fold Cross-Validation (k=5)...")

        # Performing Cross-Validation.
        # METHOD: cross_val_score(estimator, X, y, cv)
        # ARGUMENTS:
        #   - estimator: `model`. The linear regression model.
        #     WHY: This is the logic we want to test.
        #   - X: `X`. The features.
        #   - y: `y`. The target.
        #   - cv: `kf`. The splitting strategy we defined above.
        #     WHY: Defines how to chop the data up.
        # WHAT: 
        #   1. Splits data into 5 parts.
        #   2. Runs 5 separate training runs. In run 1, it uses parts 2,3,4,5 to train and part 1 to test.
        #   3. Calculates the score (default for LinearRegression is R-squared) for each run.
        # EXPECTED OUTPUT: An array `scores` containing 5 float values.
        scores = cross_val_score(model, X, y, cv=kf)

        # Outputting detailed results.
        print("\nCross-Validation Scores (R-squared) for each fold:")
        # Loop to print each score with its index.
        for i, score in enumerate(scores, 1):
            print(f"Fold {i}: {score:.4f}")

        # Calculating aggregations.
        # METHOD: np.mean(scores)
        # WHAT: The average performance.
        # WHY: A single number is easier to communicate than 5 numbers.
        mean_score = np.mean(scores)

        # METHOD: np.std(scores)
        # WHAT: The standard deviation.
        # WHY: Measures how much the score varies. If this is high, the model is unstable (fragile).
        # EXPECTED OUTPUT: Float value `std_score`.
        std_score = np.std(scores)

        print(f"\nMean R-squared Score: {mean_score:.4f}")
        print(f"Standard Deviation of Scores: {std_score:.4f}")
        
        print("\nInterpretation:")
        print(f"The model explains on average {mean_score*100:.2f}% of the variance in housing prices.")
        if std_score < 0.1:
            print("The low standard deviation suggests the model is STABLE and performs consistently across different data subsets.")
        else:
            print("The high standard deviation suggests the model is UNSTABLE and highly sensitive to the specific data points used for training.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_kfold_validation()
