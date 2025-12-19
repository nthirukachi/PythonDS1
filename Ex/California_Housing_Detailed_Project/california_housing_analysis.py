"""
# =============================================================================
# PROBLEM STATEMENT:
# =============================================================================
# The objective is to perform a comprehensive analysis of the California Housing dataset to prepare it for Machine Learning tasks.
# We need to understand the data distribution, detect and handle outliers to ensure data quality, and apply feature scaling
# to normalize the range of independent variables.
#
# STEPS TO SOLVE THE PROBLEM:
# 1.  **Data Loading**: Fetch the dataset from Scikit-Learn's repository.
# 2.  **Exploratory Data Analysis (EDA)**: Understand the data shape, types, and statistics.
# 3.  **Outlier Detection**:
#     - Sub-problem 1: Identify extreme values using Percentiles.
#     - Sub-problem 2: Detect outliers using the Interquartile Range (IQR) method.
#     - Sub-problem 3: Detect outliers using Z-Score.
#     - Sub-problem 4: Identify logical inconsistencies (e.g., Bedrooms > Rooms).
# 4.  **Data Cleaning**: Remove the identified outliers to improve model quality.
# 5.  **Feature Engineering**: 
#     - Separate Features (X) and Target (y).
#     - Identify Numerical vs Categorical columns.
# 6.  **Feature Scaling**: Apply StandardScaler to numerical features to mean=0, std=1.
#
# EXPECTED OUTPUT:
# - A clean, scaled dataset ready for training. (Mean approx 0, Std approx 1).
# - Console logs detailing the number of outliers found and removed.
# - Statistical summaries before and after scaling to verify transformation.
# =============================================================================
"""

# 2.1 Definition: Import the 'fetch_california_housing' function.
# 2.2 Why: This function provides a direct and standardized way to download the California Housing dataset.
# 2.3 When: At the start of the project when data acquisition is needed.
# 2.4 Where: Import section.
# 2.5 How to use: `from sklearn.datasets import fetch_california_housing`.
# 2.6 How it works: It downloads the data from a remote repository or loads it from a local cache.
# 2.7 Output: A function object.
from sklearn.datasets import fetch_california_housing

# 2.1 Definition: Import Pandas library.
# 2.2 Why: For data manipulation and analysis using DataFrames (tabular data structure).
# 2.3 When: Whenever we deal with structured data (rows and columns).
# 2.4 Where: Import section.
# 2.5 How to use: `import pandas as pd`.
# 2.6 How it works: Loads the library into memory and aliases it as 'pd' for convenience.
# 2.7 Output: Module object 'pd'.
import pandas as pd

# 2.1 Definition: Import NumPy library.
# 2.2 Why: For numerical operations, array handling, and mathematical functions (like mean, std, abs).
# 2.3 When: Performing mathematical computations on data.
# 2.4 Where: Import section.
# 2.5 How to use: `import numpy as np`.
# 2.6 How it works: Provides C-optimized array operations.
# 2.7 Output: Module object 'np'.
import numpy as np

# 2.1 Definition: Import SSL library.
# 2.2 Why: To handle Secure Sockets Layer (HTTPS) verification issues during dataset download.
# 2.3 When: If network security settings block the download of the dataset.
# 2.4 Where: Import section.
# 2.5 How to use: `import ssl`.
# 2.6 How it works: Modifies the default SSL context for Python's socket connections.
# 2.7 Output: Module object 'ssl'.
import ssl

# 2.1 Definition: Import Pyplot from Matplotlib.
# 2.2 Why: To generate visualizations like boxplots and histograms.
# 2.3 When: During Exploratory Data Analysis (EDA).
# 2.4 Where: Import section.
# 2.5 How to use: `import matplotlib.pyplot as plt`.
# 2.6 How it works: Provides a state-machine interface for plotting.
# 2.7 Output: Module object 'plt'.
import matplotlib.pyplot as plt

# 2.1 Definition: Import StandardScaler class.
# 2.2 Why: To standardize features by removing the mean and scaling to unit variance.
# 2.3 When: Preprocessing data for algorithms sensitive to feature scales (e.g., Regression, Neural Nets).
# 2.4 Where: Import section.
# 2.5 How to use: `from sklearn.preprocessing import StandardScaler`.
# 2.6 How it works: Implements the transformer interface (fit, transform).
# 2.7 Output: Class object 'StandardScaler'.
from sklearn.preprocessing import StandardScaler


def detect_outliers_iqr(df, column):
    """
    3.1 Argument `df`: The Pandas DataFrame containing the data.
        3.2 Why: To access the dataset.
        3.3 When: Calling the function.
        3.4 Where: Function signature.
        3.5 How to use: Pass the dataframe variable.
    3.1 Argument `column`: The name of the column to check.
        3.2 Why: To specify which feature to analyze.
        3.3 When: Calling the function.
        3.4 Where: Function signature.
        3.5 How to use: Pass string, e.g., 'AveRooms'.
    """
    # 2.1 Definition: Calculate 25th Percentile (Q1).
    # 2.2 Why: Defines the lower boundary of the "middle 50%" of data.
    # 2.3 When: Calculating IQR.
    # 2.4 Where: Inside outlier detection function.
    # 2.5 How to use: `df[column].quantile(0.25)`.
    # 2.6 How it works: Sorts data and finds the value at the 25th position.
    # 2.7 Output: Float value.
    Q1 = df[column].quantile(0.25)

    # 2.1 Definition: Calculate 75th Percentile (Q3).
    # 2.2 Why: Defines the upper boundary of the "middle 50%" of data.
    # 2.3 When: Calculating IQR.
    # 2.4 Where: Inside function.
    # 2.5 How to use: `df[column].quantile(0.75)`.
    # 2.6 How it works: Sorts data and finds value at 75% position.
    # 2.7 Output: Float value.
    Q3 = df[column].quantile(0.75)

    # 2.1 Definition: Calculate Interquartile Range (IQR).
    # 2.2 Why: Represents the spread of the middle 50% of the data.
    # 2.3 When: Determining outlier fences.
    # 2.4 Where: Inside function.
    # 2.5 How to use: `Q3 - Q1`.
    # 2.6 How it works: Subtraction.
    # 2.7 Output: Float value.
    IQR = Q3 - Q1

    # 2.1 Definition: Calculate Lower Fence.
    # 2.2 Why: Values below this are considered outliers.
    # 2.3 When: Filtering data.
    # 2.4 Where: Inside function.
    # 2.5 How to use: `Q1 - 1.5 * IQR`.
    # 2.6 How it works: Extends 1.5 times the IQR below Q1.
    # 2.7 Output: Float threshold.
    lower_bound = Q1 - 1.5 * IQR

    # 2.1 Definition: Calculate Upper Fence.
    # 2.2 Why: Values above this are considered outliers.
    # 2.3 When: Filtering data.
    # 2.4 Where: Inside function.
    # 2.5 How to use: `Q3 + 1.5 * IQR`.
    # 2.6 How it works: Extends 1.5 times the IQR above Q3.
    # 2.7 Output: Float threshold.
    upper_bound = Q3 + 1.5 * IQR

    # 2.1 Definition: Return outliers.
    # 2.2 Why: To identify rows that violate the IQR bounds.
    # 2.3 When: Returning result.
    # 2.4 Where: End of function.
    # 2.5 How to use: Boolean indexing `df[...]`.
    # 2.6 How it works: Selects rows where value < lower or value > upper.
    # 2.7 Output: Subset of DataFrame containing outliers.
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


def remove_outliers_iqr(df, column):
    """
    3.1 Argument `df`: Input DataFrame.
    3.2 Argument `column`: Column to clean.
    """
    # 2.1 Definition: Calculate Q1.
    # ... (Same logic as above) ...
    Q1 = df[column].quantile(0.25)
    
    # 2.1 Definition: Calculate Q3.
    Q3 = df[column].quantile(0.75)
    
    # 2.1 Definition: Calculate IQR.
    IQR = Q3 - Q1
    
    # 2.1 Definition: Calculate bounds.
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 2.1 Definition: Return Cleaned Data.
    # 2.2 Why: To keep only the "good" data.
    # 2.3 When: Cleaning step.
    # 2.4 Where: End of function.
    # 2.5 How to use: `(val >= lower) & (val <= upper)`.
    # 2.6 How it works: Filters for rows within the fences.
    # 2.7 Output: Cleaned DataFrame.
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def detect_outliers_zscore(df, column, threshold=3):
    """
    3.1 Argument `threshold`: Z-score limit (default 3).
        3.2 Why: Defines how many standard deviations away from mean is 'extreme'.
        3.3 When: Calling function.
    """
    # 2.1 Definition: Calculate Z-Scores.
    # 2.2 Why: Standardizes values to understand deviation from mean.
    # 2.3 When: Z-score outlier detection.
    # 2.4 Where: Inside function.
    # 2.5 How to use: `(col - mean) / std`.
    # 2.6 How it works: Subtracts mean and divides by standard deviation.
    # 2.7 Output: Series of Z-scores.
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    
    # 2.1 Definition: Filter by Threshold.
    # 2.2 Why: Finds rows where absolute Z-score exceeds limit.
    # 2.3 When: Returning result.
    # 2.4 Where: End of function.
    # 2.5 How to use: `abs(z) > threshold`.
    # 2.6 How it works: Absolute value handles both positive (high) and negative (low) outliers.
    # 2.7 Output: Outlier DataFrame.
    return df[np.abs(z_scores) > threshold]


def main():
    # 2.1 Definition: Set SSL Context.
    # 2.2 Why: Fixes potential 'certificate verify failed' errors on some systems during download.
    # 2.3 When: Before fetching data from web.
    # 2.4 Where: Main function start.
    # 2.5 How to use: `ssl._create_default... = ...`.
    # 2.6 How it works: Overrides default SSL context to allow unverified HTTPS.
    # 2.7 Output: None (Configuration change).
    ssl._create_default_https_context = ssl._create_unverified_context

    # 2.1 Definition: Fetch Dataset.
    # 2.2 Why: To load the raw data for analysis.
    # 2.3 When: Step 1.
    # 2.4 Where: Main function.
    # 2.5 How to use: `fetch_california_housing(as_frame=True)`.
    # 2.6 How it works: Downloads/Loads data and returns a Bunch object with a 'frame' attribute.
    # 2.7 Output: Bunch object containing data and target.
    california = fetch_california_housing(as_frame=True)

    # 2.1 Definition: Extract Data and Target.
    # 2.2 Why: 'data' contains features, 'target' contains labels; we need both.
    # 2.3 When: Initialization.
    # 2.4 Where: Main.
    # 2.5 How to use: Access `.data` and `.target`.
    # 2.6 How it works: Attribute access.
    # 2.7 Output: DataFrame (X) and Series (y).
    X = california.data
    y = california.target

    # 2.1 Definition: Create Working DataFrame.
    # 2.2 Why: Combining them makes row-wise operations (like filtering outliers) easier.
    # 2.3 When: Preparing DataFrame.
    # 2.4 Where: Main.
    # 2.5 How to use: `X.copy()`.
    # 2.6 How it works: Creates a deep copy of X to safely modify it.
    # 2.7 Output: New DataFrame 'df'.
    df = X.copy()
    
    # 2.1 Definition: Add Target Column.
    # 2.2 Why: To analyze relationships between features and the target.
    # 2.3 When: Data Preparation.
    # 2.4 Where: Main.
    # 2.5 How to use: `df['Name'] = values`.
    # 2.6 How it works: Assigns the series to a new column key.
    # 2.7 Output: Updated DataFrame.
    df['MedHouseValue'] = y

    # 2.1 Definition: Print Shape.
    # 2.2 Why: To know the dimensionality (Rows, Columns) of the dataset.
    # 2.3 When: EDA Step.
    # 2.4 Where: Main.
    # 2.5 How to use: `print(df.shape)`.
    # 2.6 How it works: Accesses the .shape tuple.
    # 2.7 Output: (20640, 9).
    print(f"Dataset shape: {df.shape}")

    # 2.1 Definition: Print Head.
    # 2.2 Why: To visually preview the first few records and checking format.
    # 2.3 When: EDA Step.
    # 2.4 Where: Main.
    # 2.5 How to use: `print(df.head())`.
    # 2.6 How it works: Returns top 5 rows by default.
    # 2.7 Output: Console output of 5 rows.
    print("First five rows of the dataset:")
    print(df.head())

    # 2.1 Definition: Describe Dataset.
    # 2.2 Why: To understand central tendency (mean/median) and spread (std/quartiles).
    # 2.3 When: EDA Step.
    # 2.4 Where: Main.
    # 2.5 How to use: `print(df.describe())`.
    # 2.6 How it works: Computes summary stats for numerical columns.
    # 2.7 Output: Statistical summary table.
    print("\nSummary statistics of the dataset:")
    print(df.describe())

    # 2.1 Definition: Data Info.
    # 2.2 Why: To check for missing values (Non-Null count) and data types (Dtype).
    # 2.3 When: EDA Step.
    # 2.4 Where: Main.
    # 2.5 How to use: `print(df.info())`.
    # 2.6 How it works: Summarizes metadata.
    # 2.7 Output: Info summary.
    print("\nDataFrame info:")
    print(df.info())

    # 2.1 Definition: Global Outlier Check.
    # 2.2 Why: To check validity of target variable (Negative prices or Capped maximums).
    # 2.3 When: Data Quality Check.
    # 2.4 Where: Main.
    # 2.5 How to use: boolean filtering.
    # 2.6 How it works: Checks if House Value < 0 or > 5.
    # 2.7 Output: Invalid rows.
    outliers = df[(df['MedHouseValue'] < 0) | (df['MedHouseValue'] > 5)]
    print("\nPotential outliers in 'MedHouseValue':")
    print(outliers)

    # 2.1 Definition: Check Nulls.
    # 2.2 Why: Models cannot handle Nan values usually.
    # 2.3 When: Data Cleaning.
    # 2.4 Where: Main.
    # 2.5 How to use: `df.isnull().sum()`.
    # 2.6 How it works: Returns boolean mask for nulls, then sums them up (True=1).
    # 2.7 Output: Count of nulls per column.
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # -------------------------------
    # METHOD 1: Percentiles
    # -------------------------------
    print("\n=== METHOD 1: Extreme Values (Percentiles) ===")
    # 2.1 Definition: Calculate detailed Quantiles.
    # 2.2 Why: To see the distribution's tail behavior.
    # 2.3 When: Outlier analysis.
    # 2.4 Where: Main.
    # 2.5 How to use: `df.quantile([list])`.
    # 2.6 How it works: Returns values at specified percentile cuts.
    # 2.7 Output: Series of quantiles.
    print(df['MedHouseValue'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))

    # -------------------------------
    # METHOD 2: IQR Method
    # -------------------------------
    print("\n=== METHOD 2: IQR Method ===")
    # 2.1 Definition: Interactive IQR check on Target.
    # 2.2 Why: Specifically checking target variable distribution.
    # 2.3 When: Outlier analysis.
    Q1 = df['MedHouseValue'].quantile(0.25)
    Q3 = df['MedHouseValue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 2.1 Definition: Filter Target Outliers.
    iqr_outliers = df[(df['MedHouseValue'] < lower_bound) | (df['MedHouseValue'] > upper_bound)]
    print(f"Outliers detected: {len(iqr_outliers)}")
    print(iqr_outliers)

    # -------------------------------
    # METHOD 5: Logical Inconsistencies
    # -------------------------------
    print("\n=== METHOD 5: Logical Inconsistencies ===")
    # 2.1 Definition: Logical Check.
    # 2.2 Why: Data integrity check. A house shouldn't have more bedrooms than total rooms.
    # 2.3 When: Data Validation.
    # 2.5 How to use: `df[condition]`.
    # 2.6 How it works: Comparison of two columns.
    # 2.7 Output: Rows violating logic.
    logical_issues = df[df['AveBedrms'] > df['AveRooms']]
    print(f"Records where bedrooms > rooms: {len(logical_issues)}")
    print(logical_issues)

    # -------------------------------
    # Visualization
    # -------------------------------
    # 2.1 Definition: Create Boxplots.
    # 2.2 Why: Visual confirmation of spread and outliers.
    # 2.3 When: Viz step.
    # 2.5 How to use: `df.boxplot()`.
    # 2.6 How it works: Draws box from Q1 to Q3 and whiskers to 1.5 IQR.
    # 2.7 Output: Graph window (blocking).
    plt.figure()
    df.boxplot(column=['AveRooms', 'Population'])
    plt.title("Boxplots for Outlier Detection")
    # plt.show() # Commented out to prevent blocking in automation

    # -------------------------------
    # Cleaning Pipeline
    # -------------------------------
    # 2.1 Definition: Clean 'AveRooms'.
    # 2.2 Why: Remove skew caused by mansions/hotels.
    # 2.3 When: Cleaning step.
    # 2.5 How to use: `remove_outliers_iqr`.
    # 2.6 How it works: Filters frame.
    # 2.7 Output: Cleaned Frame.
    df_cleaned = remove_outliers_iqr(df, 'AveRooms')
    
    # 2.1 Definition: Clean 'Population'.
    # 2.2 Why: Remove high leverage points (highly populated areas).
    # 2.3 When: Sequential cleaning step.
    # 2.5 How to use: Pass result of previous step.
    # 2.6 How it works: Chained filtering.
    # 2.7 Output: Further cleaned Frame.
    df_cleaned = remove_outliers_iqr(df_cleaned, 'Population')

    print("\nShape before outlier removal:", df.shape)
    print("Shape after outlier removal:", df_cleaned.shape)

    # =========================================================================
    # FEATURE SCALING
    # =========================================================================
    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)

    # 2.1 Definition: Separate Features/Target.
    # 2.2 Why: Scaling should apply to features (X) not target (y).
    # 2.3 When: Pre-processing.
    # 2.5 How to use: `drop` and selection.
    # 2.6 How it works: Splits table.
    # 2.7 Output: X dataframe, y Series.
    features = df_cleaned.drop(columns=['MedHouseValue'])
    target = df_cleaned['MedHouseValue']

    # 2.1 Definition: Identify Numerical Columns.
    # 2.2 Why: Only numerical data needs standardization.
    # 2.3 When: Pre-processing.
    # 2.5 How to use: `select_dtypes`.
    # 2.6 How it works: Checks column types.
    # 2.7 Output: List of column names.
    numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    print("\n--- APPLYING STANDARDSCALER ---")
    
    # 2.1 Definition: Initialize Scaler.
    # 2.2 Why: To prepare the transformation logic.
    # 2.3 When: Before fitting.
    # 2.5 How to use: `StandardScaler()`.
    # 2.6 How it works: Creates instance.
    # 2.7 Output: Scaler object.
    scaler = StandardScaler()
    
    # 2.1 Definition: Fit and Transform.
    # 2.2 Why: Computes mean/std and applies (x-u)/s.
    # 2.3 When: Scaling step.
    # 2.5 How to use: `fit_transform(data)`.
    # 2.6 How it works: Iterates data to find stats, then vectorizes transformation.
    # 2.7 Output: NumPy Array (not DataFrame).
    scaled_features_array = scaler.fit_transform(features[numerical_columns])

    # 2.1 Definition: Reconstruct DataFrame.
    # 2.2 Why: To get back column names and indices lost in numpy conversion.
    # 2.3 When: After scaling.
    # 2.5 How to use: `pd.DataFrame constructor`.
    # 2.6 How it works: Wraps array with metadata.
    # 2.7 Output: Scaled DataFrame.
    df_scaled = pd.DataFrame(
        scaled_features_array,
        columns=numerical_columns,
        index=features.index
    )

    # 2.1 Definition: Re-attach Target.
    # 2.2 Why: To have the final complete dataset ready for ML.
    # 2.3 When: Final step.
    # 2.5 How to use: Assignment.
    # 2.6 How it works: Adds column.
    # 2.7 Output: Final Dataset.
    df_scaled['MedHouseValue'] = target.values

    print("\n--- FINAL SCALED DATASET ---")
    print(f"Shape: {df_scaled.shape}")
    print("\nFirst 5 rows of scaled data:")
    print(df_scaled.head())
    
    # 2.1 Definition: Verify Scaling.
    # 2.2 Why: To confirm Mean is ~0 and Std is ~1.
    # 2.3 When: Validation step.
    # 2.5 How to use: `.mean()` and `.std()`.
    # 2.6 How it works: Aggregate calculation.
    # 2.7 Output: Stats series.
    print("\n--- SCALING VERIFICATION ---")
    print(f"Mean of scaled features (Should be ~0):\n{df_scaled[numerical_columns].mean()}")
    print(f"\nStd of scaled features (Should be ~1):\n{df_scaled[numerical_columns].std()}")

if __name__ == "__main__":
    main()