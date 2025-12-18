"""
PROBLEM STATEMENT:
The objective is to perform a comprehensive analysis of the California Housing dataset to prepare it for Machine Learning tasks. 
This involves fetching data, exploring its structure, identifying and handling outliers using statistical methods (IQR, Z-Score), 
and finally performing Feature Scaling (Standardization).

STEPS TO SOLVE THE PROBLEM:
1.  **Data Loading**: Fetch the dataset from Scikit-Learn's repository.
2.  **Exploratory Data Analysis (EDA)**: Understand the data shape, types, and statistics.
3.  **Outlier Detection**:
    *   Sub-problem 1: Identify extreme values using Percentiles.
    *   Sub-problem 2: Detect outliers using the Interquartile Range (IQR) method.
    *   Sub-problem 3: Detect logical inconsistencies (e.g., Bedrooms > Rooms).
    *   Sub-problem 4: check for Z-score deviations.
4.  **Data Cleaning**: Remove the identified outliers to improve model quality.
5.  **Feature Scaling**: Standardize the Cleaned data so all features contribute equally.

EXPECTED OUTPUT:
*   A summary of the dataset (shape, head, describe).
*   Lists of identified outliers from different methods.
*   Boxplots visualizing the spread and outliers.
*   Final shape of the dataset after cleaning.
*   A sample of the scaled data (Mean ~ 0, Std ~ 1).
"""

# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

# 2.1 Definition: Import the 'fetch_california_housing' function.
# 2.2 Why: This function provides a direct way to download the standard dataset.
# 2.3 When: When you need practice data for regression problems.
# 2.4 Where: At the start of the script.
# 2.5 How: `from sklearn.datasets import fetch_california_housing`
# 2.6 Works: It connects to the online repository, downloads the CSV, and caches it.
# 2.7 Output: A function object available for use.
from sklearn.datasets import fetch_california_housing

# 2.1 Definition: Import Pandas library.
# 2.2 Why: For data manipulation using DataFrames (tables).
# 2.3 When: Always when dealing with structured tabular data.
# 2.4 Where: Start of script.
# 2.5 How: `import pandas as pd`
# 2.6 Works: Loads the library into memory alias 'pd'.
# 2.7 Output: Module 'pd'.
import pandas as pd

# 2.1 Definition: Import NumPy library.
# 2.2 Why: For numerical operations (absolute values, math).
# 2.3 When: When doing math on arrays or columns.
# 2.4 Where: Start of script.
# 2.5 How: `import numpy as np`
# 2.6 Works: Optimizes C-based arrays for Python.
# 2.7 Output: Module 'np'.
import numpy as np

# 2.1 Definition: Import SSL library.
# 2.2 Why: To handle Secure Sockets Layer (HTTPS) verification.
# 2.3 When: Detecting SSL certificate errors during download (common in corporate/mac envs).
# 2.4 Where: Start of script.
# 2.5 How: `import ssl`
# 2.6 Works: Provides access to network security context settings.
# 2.7 Output: Module 'ssl'.
import ssl

# 2.1 Definition: Import Pyplot from Matplotlib.
# 2.2 Why: To generate visualizations (Boxplots).
# 2.3 When: You need to see data distribution graphically.
# 2.4 Where: Start of script.
# 2.5 How: `import matplotlib.pyplot as plt`
# 2.6 Works: Creates a state-machine environment for plotting.
# 2.7 Output: Module 'plt'.
import matplotlib.pyplot as plt

# 2.1 Definition: Import StandardScaler.
# 2.2 Why: To transform data to have Mean=0 and Variance=1.
# 2.3 When: Before feeding data to models sensitive to scale (SVM, KNN, Linear Reg).
# 2.4 Where: Start of script.
# 2.5 How: `from sklearn.preprocessing import StandardScaler`
# 2.6 Works: (x - u) / s
# 2.7 Output: A class for scaling.
from sklearn.preprocessing import StandardScaler


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_outliers_iqr(df, column):
    """
    3.1 What: Detects rows that are outliers based on IQR.
    3.2 Why: To identify data points that are statistically far from the median.
    3.3 When: The data is not necessarily normal, but has spread.
    3.4 Where: Inside data cleaning pipeline.
    3.5 How: `outliers = detect_outliers_iqr(my_df, 'Age')`
    
    Sample Example:
    Data: [1, 2, 3, 4, 100] -> 100 is outlier.
    """
    
    # 2.1 Definition: Calculate 25th Percentile (Q1).
    # 2.2 Why: Defines the lower boundary of the "middle 50%" of data.
    # 2.6 Works: Sorts data and finds value at 25% position.
    Q1 = df[column].quantile(0.25)

    # 2.1 Definition: Calculate 75th Percentile (Q3).
    # 2.2 Why: Defines the upper boundary of the "middle 50%".
    Q3 = df[column].quantile(0.75)

    # 2.1 Definition: Calculate Interquartile Range (IQR).
    # 2.2 Why: Represents the spread of the middle 50% of data.
    # 2.6 Works: Arithmetic subtraction Q3 - Q1.
    IQR = Q3 - Q1

    # 2.1 Definition: Calculate Lower Fence.
    # 2.2 Why: Any value below this is considered "Too Low".
    # 2.6 Works: Q1 - 1.5 * IQR (Standard statistical rule).
    lower_bound = Q1 - 1.5 * IQR

    # 2.1 Definition: Calculate Upper Fence.
    # 2.2 Why: Any value above this is "Too High".
    # 2.6 Works: Q3 + 1.5 * IQR.
    upper_bound = Q3 + 1.5 * IQR

    # 2.1 Definition: Boolean Indexing/Filtering.
    # 2.2 Why: To select only values outside the bounds.
    # 2.7 Output: A DataFrame containing only outlier rows.
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def remove_outliers_iqr(df, column):
    """
    3.1 What: Filters out rows that are outliers.
    3.2 Why: To clean the dataset.
    3.3 When: You want to proceed with analysis on "Normal" data only.
    3.5 How: `cleaned_df = remove_outliers_iqr(df, 'Price')`
    """
    # (Logic repeats calculating bounds)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 2.1 Definition: Filter for 'Good' data.
    # 2.2 Why: Removing outliers means keeping Inliers.
    # 2.6 Works: Uses '&' (AND) condition: (>= Lower AND <= Upper).
    # 2.7 Output: DataFrame with outliers removed.
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def detect_outliers_zscore(df, column, threshold=3):
    """
    3.1 What: Detects outliers using Z-Score (Standard Deviation).
    3.2 Why: Good for Normally Distributed data.
    3.3 When: You want to find how many Standard Deviations away a point is.
    3.5 How: `outliers = detect_outliers_zscore(df, 'Height')`
    
    Arguments:
    - threshold=3: 99.7% of data usually lies within 3 SDs. Anything beyond is rare.
    """
    # 2.1 Definition: Calculate Z-Score formula.
    # 2.2 Why: Normalizes the deviation.
    # 2.6 Works: (Value - Mean) / StdDev.
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    
    # 2.1 Definition: Filter by absolute threshold.
    # 2.2 Why: Checks both negative (very low) and positive (very high) deviations.
    return df[np.abs(z_scores) > threshold]


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    # 2.1 Definition: Set SSL Context to Unverified.
    # 2.2 Why: To bypass "Certificate Verify Failed" errors often seen when downloading from Python scripts.
    # 2.3 When: Running scripts that fetch https data behind proxies or firewalls.
    # 2.6 Works: Monkey-patches the default https context.
    ssl._create_default_https_context = ssl._create_unverified_context

    # 2.1 Definition: Fetch Dataset.
    # 3.1 Argument: as_frame=True (Returns a Pandas DataFrame instead of numpy array).
    # 2.2 Why: To start the analysis.
    # 2.7 Output: A Bunch object containing 'data' and 'target'.
    california = fetch_california_housing(as_frame=True)

    # 2.1 Definition: Extract Data and Target.
    # 2.2 Why: Scikit-learn separates them; we typically analyze them together.
    X = california.data
    y = california.target

    # 2.1 Definition: Merge into one DataFrame.
    # 2.2 Why: Easier to do correlation/logic checks if X and y are in the same table.
    # 2.6 Works: .copy() ensures we don't mutate the original source data structure.
    df = X.copy()
    df['MedHouseValue'] = y

    # 2.1 Definition: Print Shape.
    # 2.2 Why: To know how many rows (N) and columns (features) we have.
    # 2.7 Output: (20640, 9).
    print(f"Dataset shape: {df.shape}")

    # 2.1 Definition: Print Head.
    # 2.2 Why: To sanity-check that data loaded correctly and look at the values.
    # 2.7 Output: First 5 rows of the table.
    print("First five rows of the dataset:")
    print(df.head())

    # 2.1 Definition: Describe.
    # 2.2 Why: Returns Count, Mean, Std, Min, 25%, 50%, 75%, Max for every column.
    # 2.3 When: First step of EDA.
    print("\nSummary statistics of the dataset:")
    print(df.describe())

    # 2.1 Definition: Info.
    # 2.2 Why: To check data types (float vs object) and missing values (Non-Null count).
    print("\nDataFrame info:")
    print(df.info())

    # ---------------------------------------------------------
    # EXPLORATION: Value Analysis
    # ---------------------------------------------------------
    
    # 2.1 Definition: Check Logical Extreme Outliers.
    # 2.2 Why: Houses < 0 dollars or > 5 (which is 500k, the cap) might be issues.
    # 2.7 Output: Subset of rows matching condition.
    outliers = df[(df['MedHouseValue'] < 0) | (df['MedHouseValue'] > 5)]
    print("\nPotential outliers in 'MedHouseValue':")
    print(outliers)

    # 2.1 Definition: Check Nulls.
    # 2.2 Why: Machine Learning models crash on Nulls.
    # 2.7 Output: Sum of nulls per column.
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # METHOD 1: Percentiles
    # ---------------------------------------------------------
    print("\n=== METHOD 1: Extreme Values (Percentiles) ===")
    
    # 2.1 Definition: Calculate Quantiles at 1%, 25%, 50%, 75%, 99%.
    # 2.2 Why: To see the "tails" of the distribution.
    # 2.6 Works: Sorts and interpolates.
    print(df['MedHouseValue'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))

    # ---------------------------------------------------------
    # METHOD 2: IQR Method
    # ---------------------------------------------------------
    print("\n=== METHOD 2: IQR Method ===")
    # See helper function 'detect_outliers_iqr' for logic details.
    
    Q1 = df['MedHouseValue'].quantile(0.25)
    Q3 = df['MedHouseValue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 2.1 Definition: Find Price Outliers.
    iqr_outliers = df[(df['MedHouseValue'] < lower_bound) | (df['MedHouseValue'] > upper_bound)]
    print(f"Outliers detected: {len(iqr_outliers)}")
    print(iqr_outliers.head())

    # ---------------------------------------------------------
    # METHOD 3: Zero/Negative Checks
    # ---------------------------------------------------------
    print("\n=== METHOD 3: Zero or Negative Values ===")
    
    # 2.1 Definition: Check impossible physical values.
    # 2.2 Why: Income, Age, Population cannot be <= 0.
    zero_negative = df[(df['MedInc'] <= 0) | (df['HouseAge'] < 0) | (df['Population'] <= 0)]
    print(f"Zero/Negative values found: {len(zero_negative)}")
    print(zero_negative)

    # ---------------------------------------------------------
    # METHOD 4: Capped Values
    # ---------------------------------------------------------
    print("\n=== METHOD 4: Capped/Repeated Maximum Values ===")
    
    # 2.1 Definition: Check for artificial ceiling.
    # 2.2 Why: In this dataset, values > 500k are clipped to 500k (5.0). This creates a spike at 5.0.
    # 2.6 Works: Compares value to column max.
    capped = df[df['MedHouseValue'] == df['MedHouseValue'].max()]
    print(f"Rows with maximum value (5.0): {len(capped)}")
    # Expected Output: A large number (~965), confirming the clip.
    print(capped.head())

    # ---------------------------------------------------------
    # METHOD 5: Logical Inconsistencies
    # ---------------------------------------------------------
    print("\n=== METHOD 5: Logical Inconsistencies ===")
    
    # 2.1 Definition: Cross-column check.
    # 2.2 Why: A house cannot have more Bedrooms than total Rooms.
    logical_issues = df[df['AveBedrms'] > df['AveRooms']]
    print(f"Records where bedrooms > rooms: {len(logical_issues)}")
    print(logical_issues)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    # 2.1 Definition: Setup Figure.
    plt.figure()
    
    # 2.1 Definition: Boxplot.
    # 2.2 Why: Best visual for IQR-based outliers.
    # 2.6 Works: Draws Q1, Q3 box, and whiskers at 1.5*IQR. Dots are outliers.
    df.boxplot(column=['AveRooms', 'Population'])
    plt.title("Boxplots for Outlier Detection")
    # plt.show() # Commented for non-interactive run

    # ---------------------------------------------------------
    # CLEANING PIPELINE
    # ---------------------------------------------------------
    
    # 2.1 Definition: Detect outliers in specific columns.
    ave_rooms_outliers = detect_outliers_iqr(df, 'AveRooms')
    population_outliers = detect_outliers_iqr(df, 'Population')

    print("\nNumber of AveRooms outliers (IQR):", len(ave_rooms_outliers))
    print("Number of Population outliers (IQR):", len(population_outliers))

    detect_outliers_zscore(df, 'Population')

    # 2.1 Definition: Apply removal.
    # 2.2 Why: To create the 'Gold Standard' dataset for training.
    df_cleaned = remove_outliers_iqr(df, 'AveRooms')
    
    # 2.1 Definition: Sequential cleaning.
    # 2.6 Works: We clean Population FROM the dataset that already had AveRooms cleaned.
    df_cleaned = remove_outliers_iqr(df_cleaned, 'Population')

    print("\nShape before outlier removal:", df.shape)
    print("Shape after outlier removal:", df_cleaned.shape)

    # ---------------------------------------------------------
    # FEATURE SCALING
    # ---------------------------------------------------------
    
    # 2.1 Definition: Drop Target.
    # 2.2 Why: We only scale features (X), never the target (y) usually (unless transforming dist).
    features = df_cleaned.drop(columns=['MedHouseValue'])
    target = df_cleaned['MedHouseValue']

    # 2.1 Definition: Initialize Scaler.
    scaler = StandardScaler()
    
    # 2.1 Definition: Fit and Transform.
    # 2.2 Why: 'Fit' calculates Mean/Std. 'Transform' applies math.
    # 2.7 Output: Numpy Array of scaled values.
    scaled_features = scaler.fit_transform(features)

    # 2.1 Definition: Reconstruct DataFrame.
    # 2.2 Why: Scaler returns array, we want DataFrame for readability.
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    
    # 2.1 Definition: Add Target back.
    df_scaled['MedHouseValue'] = target.values

    print("\nScaled feature sample:")
    # Expected Output: Values like -0.5, 1.2 (small floats centered around 0).
    print(df_scaled.head())

if __name__ == "__main__":
    main()