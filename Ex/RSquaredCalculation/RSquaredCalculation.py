"""
Problem Statement:
Using the housing prices dataset, Write Python code to calculate R-squared and Adjusted R-squared values for a regression model.
R-squared explains the proportion of variance in the dependent variable that is predictable from the independent variables.
Adjusted R-squared penalizes adding unnecessary features that do not improve the model.

Steps to Solve the Problem:
1.  Import necessary libraries: pandas (data), sklearn (model & metrics).
2.  Load the dataset ('Housing.csv').
3.  Preprocess the data:
    -   Convert categorical variables to numeric (One-Hot Encoding).
    -   Separate features (X) and target variable (y - 'price').
4.  Split data into Training and Testing sets.
5.  Initialize and Train a Linear Regression model.
6.  Make predictions on the test set.
7.  Calculate R-squared score using sklearn's metric.
8.  Calculate Adjusted R-squared using the formula: 1 - (1 - R2) * ( (n - 1) / (n - p - 1) ).
    -   n: number of samples.
    -   p: number of predictors.
9.  Output the results.

Sub-problems:
-   Data preparation (Encoding).
-   Model Fitting.
-   Metric computation.

Expected Output:
-   Printed R-squared value (e.g., 0.65).
-   Printed Adjusted R-squared value (e.g., 0.63).
"""

# Importing pandas.
# WHAT: Library for data manipulation.
# WHY: To load and structure the housing data.
# WHEN: Always at the beginning of a data script.
# EXPECTED OUTPUT: Module `pd` is loaded.
import pandas as pd

# Importing LinearRegression.
# WHAT: The standard class for Ordinary Least Squares Regression.
# WHY: We need a model to fit the data before we can check its performance (R-squared).
# WHEN: You want to predict a continuous variable (price).
# EXPECTED OUTPUT: `LinearRegression` class is available.
from sklearn.linear_model import LinearRegression

# Importing train_test_split.
# WHAT: Function to split arrays or matrices into random train and test subsets.
# WHY: To evaluate the model on unseen data, preventing overfitting.
# WHEN: Before training a machine learning model.
# EXPECTED OUTPUT: `train_test_split` function is available.
from sklearn.model_selection import train_test_split

# Importing r2_score.
# WHAT: Regression score function.
# WHY: To calculate the coefficient of determination (R^2).
# WHEN: Evaluating regression model performance.
# EXPECTED OUTPUT: `r2_score` function is available.
from sklearn.metrics import r2_score

def calculate_r_squared_metrics():
    # Defining file path.
    # WHAT: String literal for file location.
    # WHY: Single source of truth for the file name.
    # EXPECTED OUTPUT: String 'Housing.csv'.
    file_path = 'Housing.csv'

    try:
        # Loading data.
        # METHOD: pd.read_csv(filepath)
        # ARGUMENTS:
        #   - filepath: 'Housing.csv'.
        #     WHY: Specifies source file.
        # WHAT: Reads CSV into DataFrame.
        # EXPECTED OUTPUT: DataFrame `df`.
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        # Data Preprocessing: Handling Categorical Data.
        # WHAT: Convert text categories to numbers.
        # WHY: Linear Regression requires numerical input.
        # METHOD: pd.get_dummies(data, drop_first)
        # ARGUMENTS:
        #   - data: `df`. Input data.
        #   - drop_first: True. 
        #     WHY: To prevent multicollinearity (dummy variable trap).
        #     WHEN: Using linear models.
        # EXPECTED OUTPUT: `df_numeric` with only number columns.
        df_numeric = pd.get_dummies(df, drop_first=True)

        # Separating Target and Features.
        # WHAT: Defining 'y' as the output we want to predict (price).
        # WHY: Supervised learning requires a defined target.
        # EXPECTED OUTPUT: Series `y` containing prices.
        y = df_numeric['price']

        # WHAT: Defining 'X' as the input features (everything except price).
        # METHOD: df.drop(columns, axis)
        # ARGUMENTS:
        #   - columns: 'price'. The label to remove.
        #   - axis: 1. Denotes columns (not rows).
        # EXPECTED OUTPUT: DataFrame `X` without the price column.
        X = df_numeric.drop('price', axis=1)

        # Splitting the data.
        # METHOD: train_test_split(*arrays, test_size, random_state)
        # ARGUMENTS:
        #   - *arrays: X, y. The inputs to be split.
        #   - test_size: 0.2.
        #     WHY: 20% of data is kept for testing, 80% for training.
        #     WHEN: Standard practice is 70-30 or 80-20.
        #   - random_state: 42.
        #     WHY: To act as a seed for the random number generator, ensuring reproducibility.
        #     WHEN: You want the same split every time you run the code.
        # WHAT: Returns 4 arrays/matrices: X_train, X_test, y_train, y_test.
        # EXPECTED OUTPUT: Four arrays with split data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initializing the Model.
        # METHOD: LinearRegression()
        # WHAT: Creates an instance of the model.
        # EXPECTED OUTPUT: An empty model object `model`.
        model = LinearRegression()

        # Training the Model.
        # METHOD: model.fit(X, y)
        # ARGUMENTS:
        #   - X: `X_train`. The training features.
        #   - y: `y_train`. The training target.
        # what: Computes the coefficients (weights) for the linear equation.
        # WHY: To 'learn' the relationship between features and price.
        # EXPECTED OUTPUT: The model object is now trained.
        model.fit(X_train, y_train)

        # Making Predictions.
        # METHOD: model.predict(X)
        # ARGUMENTS:
        #   - X: `X_test`. The unseen test features.
        #     WHY: To see how well the model guesses prices it hasn't seen before.
        # WHAT: Generates predicted prices.
        # EXPECTED OUTPUT: Array `y_pred` containing predicted values.
        y_pred = model.predict(X_test)

        # Calculating R-squared.
        # METHOD: r2_score(y_true, y_pred)
        # ARGUMENTS:
        #   - y_true: `y_test`. The actual prices.
        #   - y_pred: `y_pred`. The modeled prices.
        #     WHY: R2 compares the errors of the model against the errors of a simple mean baseline.
        # WHAT: Returns a float score (usually 0 to 1). 1 is perfect prediction.
        # EXPECTED OUTPUT: Float variable `r2`.
        r2 = r2_score(y_test, y_pred)

        print(f"\nR-squared: {r2:.4f}")
        print("-> Interpretation: This represents the percentage of variance in Price explained by the features.")

        # Calculating Adjusted R-squared.
        # WHAT: A modified version of R-squared that adjusts for the number of predictors.
        # FORMULA: 1 - (1-R2)*(n-1) / (n-p-1)
        # WHY: Standard R-squared allows increases even if you add junk features. Adjusted R-squared penalizes that.
        # WHEN: Comparing models with different numbers of predictors.
        
        # 'n' is the number of samples in the test set.
        n = len(y_test)
        
        # 'p' is the number of features (columns in X).
        p = X_test.shape[1]

        # Computing the value.
        # STEPS:
        # 1. Calculate (1 - r2).
        # 2. Calculate (n - 1).
        # 3. Calculate (n - p - 1). (Degrees of freedom of the error).
        # 4. Multiply and subtract from 1.
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        print(f"Adjusted R-squared: {adj_r2:.4f}")
        print("-> Interpretation: This penalizes the score if unnecessary features are added.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Hint for user in case of common error with dummy data
        if "division by zero" in str(e) or "float division by zero" in str(e):
             print("\nNOTE: Division by zero usually happens if n <= p + 1. The dataset might be too small for the number of features created.")

if __name__ == "__main__":
    calculate_r_squared_metrics()
