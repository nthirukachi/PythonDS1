"""
Problem Statement:
Using the housing prices dataset, Explain Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error with examples from your housing prices prediction.

Steps to Solve the Problem:
1.  Import libraries: pandas (data), sklearn (model & metrics), numpy (math).
2.  Load the dataset. Since this script is in a subfolder, we look one level up ('../Housing.csv').
3.  Preprocess the data:
    -   Convert categorical features to numeric (One-Hot Encoding).
    -   Separate X (features) and y (target/price).
4.  Split data into Training and Testing sets.
5.  Train a Linear Regression model.
6.  Make predictions on the Test set.
7.  Calculate Mean Absolute Error (MAE): Average of absolute errors.
8.  Calculate Mean Squared Error (MSE): Average of squared errors.
9.  Calculate Root Mean Squared Error (RMSE): Square root of MSE.
10. Output the values with detailed interpretation.

Sub-problems:
-   Data loading from relative path.
-   Model training.
-   Metric computation.

Expected Output:
-   Three float values: MAE, MSE, RMSE.
-   Interpretation text explaining what each value implies about the prediction error in dollars.
"""

# Importing pandas.
# WHAT: Library for data manipulation.
# WHY: to store data in a DataFrame.
# EXPECTED OUTPUT: `pd` module.
import pandas as pd

# Importing numpy.
# WHAT: Scientific computing library.
# WHY: We need `np.sqrt()` to calculate RMSE from MSE.
# WHEN: Performing mathematical transformations.
# EXPECTED OUTPUT: `np` module.
import numpy as np

# Importing LinearRegression.
# WHAT: The model class.
# WHY: To fit a line to the data.
# EXPECTED OUTPUT: `LinearRegression` class.
from sklearn.linear_model import LinearRegression

# Importing train_test_split.
# WHAT: Validation tool.
# WHY: To check error on unseen data.
# EXPECTED OUTPUT: function available.
from sklearn.model_selection import train_test_split

# Importing metrics.
# WHAT: Collection of score functions.
# WHY: Specifically `mean_absolute_error` and `mean_squared_error`.
# WHEN: Evaluating model accuracy.
# EXPECTED OUTPUT: metric functions available.
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_error_metrics():
    # Defining file path.
    # WHAT: Relative path to the parent directory.
    # WHY: The script is in `Ex/ErrorMetricsCalculation/` but data is in `Ex/`.
    # EXPECTED OUTPUT: String path.
    file_path = '../Housing.csv'

    try:
        # Loading data.
        # METHOD: pd.read_csv()
        # WHAT: loads data.
        # EXPECTED OUTPUT: DataFrame `df`.
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        # Data Preprocessing.
        # METHOD: pd.get_dummies()
        # WHAT: Converts text to numbers.
        # WHY: Regression needs math inputs.
        # EXPECTED OUTPUT: Numeric DataFrame.
        df_numeric = pd.get_dummies(df, drop_first=True)

        # Separating Target and Features.
        y = df_numeric['price']
        X = df_numeric.drop('price', axis=1)

        # Splitting Data.
        # METHOD: train_test_split()
        # ARGUMENTS: test_size=0.2 (20% test), random_state=42.
        # WHY: Standard validation split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training Model.
        # METHOD: fit()
        # WHAT: Trains the model.
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Making Predictions.
        # METHOD: predict()
        # WHAT: Generates predicted prices for the test set.
        # EXPECTED OUTPUT: Array of predicted prices `y_pred`.
        y_pred = model.predict(X_test)

        print("\n--- Error Metric Analysis ---")

        # 1. Mean Absolute Error (MAE)
        # METHOD: mean_absolute_error(y_true, y_pred)
        # ARGUMENTS:
        #   - y_true: `y_test`. Actual prices.
        #   - y_pred: `y_pred`. Predicted prices.
        # WHAT: Calculates the average absolute difference: mean(|y_true - y_pred|).
        # WHY: To see the 'average' error magnitude. It is robust to outliers (doesn't square the error).
        # EXPECTED OUTPUT: Float value.
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\nMean Absolute Error (MAE): {mae:,.2f}")
        print(f"-> Interpretation: On average, our prediction is off by ${mae:,.2f}.")

        # 2. Mean Squared Error (MSE)
        # METHOD: mean_squared_error(y_true, y_pred)
        # WHAT: Calculates the average squared difference: mean((y_true - y_pred)^2).
        # WHY: Penalizes large errors heavily. Useful during training (gradient descent) but hard to interpret (units are 'dollars squared').
        # EXPECTED OUTPUT: Very large float value.
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nMean Squared Error (MSE): {mse:,.2f}")
        print("-> Interpretation: The average squared difference. Hard to interpret directly in dollar terms.")

        # 3. Root Mean Squared Error (RMSE)
        # METHOD: np.sqrt(mse)
        # ARGUMENTS: `mse` value calculated above.
        # WHAT: Square root of MSE.
        # WHY: To bring the units back to 'dollars'. This is the standard metric. 
        #      Because it comes from MSE, it still penalizes large errors more than MAE.
        # EXPECTED OUTPUT: Float value similar scale to MAE (usually larger).
        rmse = np.sqrt(mse)
        print(f"\nRoot Mean Squared Error (RMSE): {rmse:,.2f}")
        print(f"-> Interpretation: A more popular measure. It tells us that typical errors are around ${rmse:,.2f}.")
        
        # Comparison
        print(f"\nComparison: RMSE ({rmse:,.0f}) is usually > MAE ({mae:,.0f}) because RMSE penalizes outliers more.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_error_metrics()
