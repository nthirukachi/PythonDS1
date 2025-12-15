# =====================================================================
# COLUMN TRANSFORMER WITH PREPROCESSING PIPELINE
# Applying different transformations to different column types
# =====================================================================

# Import pandas library for working with DataFrames
import pandas as pd

# Import fetch_california_housing to load the dataset
from sklearn.datasets import fetch_california_housing

# Import StandardScaler to normalize numerical features (mean=0, std=1)
# StandardScaler is essential for distance-based and gradient descent algorithms
from sklearn.preprocessing import StandardScaler

# Import OneHotEncoder to convert categorical variables to binary columns
# OneHotEncoder transforms text categories into numerical format
from sklearn.preprocessing import OneHotEncoder

# Import ColumnTransformer to apply different transformations to different columns
# ColumnTransformer allows us to preprocess numerical and categorical columns separately
# This is more efficient and Pythonic than manual preprocessing
from sklearn.compose import ColumnTransformer

# =====================================================================
# STEP 1: LOAD THE CALIFORNIA HOUSING DATASET
# =====================================================================

# Fetch the California housing dataset from scikit-learn
# as_frame=True returns a dictionary with 'data' and 'target' as DataFrames
data = fetch_california_housing(as_frame=True)

# Extract the features DataFrame from the fetched data dictionary
# data.frame contains both features (X) and target (y) combined
df = data.frame

# Display the original shape of the dataset
# Shows number of rows and columns before any preprocessing
print("Original shape:", df.shape)
# Output: Original shape: (20640, 9) - 20,640 rows, 9 columns (8 features + 1 target)

# Display detailed information about the DataFrame
# Shows data types, non-null counts, and memory usage for each column
# Helps identify which columns are numerical vs categorical
print(df.info())
# Output shows: float64 columns (numerical), object columns (categorical)

# =====================================================================
# STEP 2: INSPECT DATASET COLUMNS
# =====================================================================

# Print a blank line for better readability in console output
print("\nColumns in the dataset:")

# Display all column names in the dataset
# Helps understand what features are available for preprocessing
print(df.columns)
# Output: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseValue

# =====================================================================
# STEP 3: CREATE CATEGORICAL FEATURE FOR DEMONSTRATION
# =====================================================================

# Create a new categorical column 'IncomeCategory' from numerical 'MedInc'
# pd.cut() converts continuous numerical values into discrete categorical bins
# This demonstrates how to handle categorical data in preprocessing
# Parameters:
#   - df['MedInc']: the column to cut into bins
#   - bins: boundaries for categories [0, 2, 4, 6, 15]
#   - labels: names for each bin ['Low', 'Medium', 'High', 'VeryHigh']
df['IncomeCategory'] = pd.cut(
    df['MedInc'],
    bins=[0, 2, 4, 6, 15],
    labels=['Low', 'Medium', 'High', 'VeryHigh']
)

# Display updated DataFrame info with the new categorical column
# Now we have a mix of numerical and categorical columns
# This is needed to demonstrate ColumnTransformer's power
print(df.info())
# Output shows 'IncomeCategory' as object (categorical) type

# =====================================================================
# STEP 4: SEPARATE FEATURES FROM TARGET VARIABLE
# =====================================================================

# Drop the target column 'MedHouseVal' to get only feature columns
# We separate features (X) from target (y) for model training
# axis=1 means drop columns (axis=0 would drop rows)
# The target variable should NOT be preprocessed with features
X = df.drop('MedHouseVal', axis=1)

# Now X contains: MedInc, HouseAge, AveRooms, etc., and IncomeCategory
# But NOT the target variable MedHouseValue

# =====================================================================
# STEP 5: IDENTIFY AND CATEGORIZE COLUMNS BY DATA TYPE
# =====================================================================

# Select only numerical columns from features (float64 and int64 types)
# select_dtypes() automatically detects column data types
# These columns need StandardScaler (numerical preprocessing)
# StandardScaler: (X - mean) / standard_deviation
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
# Result: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Select only categorical columns from features (object and category types)
# These columns need OneHotEncoder (categorical preprocessing)
# OneHotEncoder converts categories to binary dummy variables
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
# Result: ['IncomeCategory']

# =====================================================================
# STEP 6: CREATE PREPROCESSING PIPELINE USING COLUMNSTRANSFORMER
# =====================================================================

# Create a ColumnTransformer object to handle preprocessing
# ColumnTransformer allows different transformations for different column types
# This is much cleaner than manually preprocessing each column type
# Why use ColumnTransformer:
#   1. Applies transformations only to specified columns
#   2. Learns scaling/encoding parameters from training data (prevents data leakage)
#   3. Can be used in sklearn pipelines with models
#   4. Handles train-test data consistently
preprocessor = ColumnTransformer(
    # transformers: list of tuples (name, transformer, columns)
    # Each tuple specifies: (unique_name, preprocessing_method, which_columns_to_apply_to)
    transformers=[
        # TUPLE 1: Numerical preprocessing
        # 'num' = unique identifier for this transformation
        # StandardScaler() = the preprocessing method
        # numerical_cols = list of columns to apply StandardScaler to
        # Purpose: Scale numerical features to mean=0, std=1
        ('num', StandardScaler(), numerical_cols),
        
        # TUPLE 2: Categorical preprocessing
        # 'cat' = unique identifier for this transformation
        # OneHotEncoder(handle_unknown='ignore') = the preprocessing method
        # handle_unknown='ignore' = if new categories appear, ignore them gracefully
        # categorical_cols = list of columns to apply OneHotEncoder to
        # Purpose: Convert categorical text to binary numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# =====================================================================
# STEP 7: APPLY PREPROCESSING TO THE FEATURES
# =====================================================================

# fit_transform() does two operations:
# 1. fit() - learns preprocessing parameters from the data
#    - For StandardScaler: learns mean and std of numerical columns
#    - For OneHotEncoder: learns all unique categories in categorical columns
# 2. transform() - applies the learned transformations
#    - For StandardScaler: scales each value using learned mean/std
#    - For OneHotEncoder: converts categories to binary columns
# X_processed is the final preprocessed feature matrix
# All numerical features are scaled, all categorical features are encoded
X_processed = preprocessor.fit_transform(X)

# =====================================================================
# STEP 8: DISPLAY THE RESULT
# =====================================================================

# Display the shape of the preprocessed data
# Shows how many rows and columns after preprocessing
# Note: shape may increase due to OneHotEncoding creating multiple columns
print("Shape after preprocessing:", X_processed.shape)
# Example output: Shape after preprocessing: (20640, 12)
# Why changed from 9 to 12?
#   - 8 numerical columns scaled (no change in count)
#   - 1 categorical column with 4 categories → 4 binary columns
#   - Total: 8 + 4 = 12 columns

# =====================================================================
# WHY USE COLUMNTRANSFORMER?
# =====================================================================
"""
ADVANTAGES OF COLUMNTRANSFORMER:

1. **Selective Transformation**
   - Only transforms specified columns
   - Leaves other columns unchanged
   - Prevents accidental transformation of wrong columns

2. **Data Consistency**
   - Learns parameters (mean, std, categories) from training data ONLY
   - Applies same parameters to test data
   - Prevents data leakage

3. **Pipeline Integration**
   - Works seamlessly with sklearn pipelines
   - Example: Pipeline([preprocessor, LogisticRegression()])
   - Makes code more modular and maintainable

4. **Handles Multiple Data Types**
   - Different transformations for different column types in one step
   - Cleaner code than manual preprocessing
   - Easier to modify preprocessing steps

5. **Reproducibility**
   - Same transformation applied consistently
   - Easier to debug and understand
   - Can save and load preprocessor for future use

COMPARISON:

Without ColumnTransformer (Manual Approach):
   - Scale numerical columns manually
   - Encode categorical columns manually
   - Keep track of column order manually
   - Error-prone and hard to maintain

With ColumnTransformer (Recommended):
   - All preprocessing in one object
   - Automatic column handling
   - Easy to use with models
   - Professional and maintainable code
"""
