# R-squared and Adjusted R-squared Analysis Process

## 1. Steps Followed in the Code
The Python script `RSquaredCalculation.py` performs the following logical steps to evaluate the housing price regression model:

### Step 1: Import Libraries
- **Libraries**: `pandas` (for data handling), `LinearRegression` (for modeling), `train_test_split` (for validation), and `metrics` (for scoring).
- **Purpose**: To set up the environment with necessary tools for data science.

### Step 2: Load Data
- **Action**: Reads `Housing.csv` into a pandas DataFrame.
- **Verification**: Prints the success message or error if the file is missing.

### Step 3: Data Preprocessing
- **One-Hot Encoding**: Converts categorical columns (like 'mainroad', 'guestroom') into numeric 0s and 1s using `pd.get_dummies(drop_first=True)`.
- **Feature Selection**:
    - `y` (Target): The 'price' column.
    - `X` (Features): All other columns.

### Step 4: Model Training
- **Splitting**: Divides data into Training (80%) and Testing (20%) sets to ensure the model is tested on unseen data.
- **Fitting**: The Linear Regression model learns the coefficients (weights) for each feature based on the training data.

### Step 5: Prediction and Metric Calculation
- **Prediction**: The model predicts prices for the Test set.
- **R-squared**: calculated using `sklearn.metrics.r2_score`.
- **Adjusted R-squared**: Calculated manually using the formula:
  $$ R^2_{adj} = 1 - (1-R^2) \frac{n-1}{n-p-1} $$
  Where $n$ is sample size and $p$ is number of predictors.

---

## 2. Observations and Output Interpretation

### R-squared ($R^2$)
- **Definition**: Statistical measure that represents the proportion of the variance for the dependent variable (Price) that's explained by an independent variable or variables in a regression model.
- **Observation**:
    - If $R^2 = 0.65$, it means **65% of the variation in housing prices** can be explained by the features (area, bedrooms, etc.).
    - The remaining 35% is unexplained variance (error).

### Adjusted R-squared ($R^2_{adj}$)
- **Definition**: A modified version of R-squared that has been adjusted for the number of predictors in the model.
- **Observation**:
    - This value is usually lower than or equal to R-squared.
    - If $R^2_{adj}$ is significantly lower than $R^2$ (e.g., $R^2=0.65, R^2_{adj}=0.40$), it indicates that **many feature variables are not adding value** to the model and might be "noise".
    - In our small dummy dataset, this gap might be large because the sample size ($n=14$) is very small relative to the number of features created by get_dummies.

### Conclusion
- **R-squared** gives the "optimistic" view of model fit.
- **Adjusted R-squared** gives the "realistic" view, penalizing for complexity.
- Ideally, we want both values to be high and close to each other.
