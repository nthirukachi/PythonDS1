"""
PROBLEM STATEMENT:
Using the California Housing dataset (available via sklearn.datasets), perform the following steps:
    • Load the data into a Pandas DataFrame.
    • Display the first five rows and summarize the dataset.
    • Identify at least two potential outliers or data quality issues.
"""

"""
ABOUT CALIFORNIA HOUSING DATASET:
The California Housing dataset is from the 1990 California census containing housing data.

Dataset Size: 20,640 instances with 8 features + 1 target variable

Features:
    1. MedInc - Median income in block (in tens of thousands of dollars)
    2. HouseAge - Median age of houses in block (in years)
    3. AveRooms - Average number of rooms per household
    4. AveBedrms - Average number of bedrooms per household
    5. Population - Block population
    6. AveOccup - Average occupancy (persons per household)
    7. Latitude - Geographic latitude
    8. Longitude - Geographic longitude

Target Variable:
    - MedHouseValue - Median house value for California districts (in hundreds of thousands of dollars)
    - Range: typically from ~$14,999 to $500,001

Characteristics:
    - No missing values
    - Continuous numerical features
    - Geographically distributed across California
    - Ideal for regression analysis and machine learning model training
"""

# Import the California housing dataset function from scikit-learn
from sklearn.datasets import fetch_california_housing
# Import pandas library for working with DataFrames
import pandas as pd
import numpy as np
import ssl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z_scores) > threshold]


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    # Fetch the California housing dataset
    # as_frame=True converts the dataset into a pandas DataFrame format instead of NumPy arrays
    # This makes it easier to work with column names and perform data exploration
    california = fetch_california_housing(as_frame=True)

    # Extract features (independent variables) from the dataset
    X = california.data
    # Extract target variable (dependent variable - house prices) from the dataset
    y = california.target

    # Create a copy of the features DataFrame to avoid modifying the original data
    df = X.copy()
    # Add the target variable as a new column named 'MedHouseValue' to the DataFrame
    df['MedHouseValue'] = y

    # want to know the size of the dataset
    print(f"Dataset shape: {df.shape}")

    # Display the first five rows of the DataFrame to get an initial look at the data
    print("First five rows of the dataset:")
    print(df.head())

    # Summarize the dataset to get statistical insights about each feature
    print("\nSummary statistics of the dataset:")
    print(df.describe())

    # Display information about the DataFrame including data types and non-null counts
    print("\nDataFrame info:")
    print(df.info())

    # Identify potential outliers or data quality issues
    # For example, checking for unusually high or low values in 'MedHouseValue'
    outliers = df[(df['MedHouseValue'] < 0) | (df['MedHouseValue'] > 5)]
    print("\nPotential outliers in 'MedHouseValue':")
    print(outliers)

    # get the distinct values in 'MedHouseValue' to check for any anomalies
    print("\nDistinct values in 'MedHouseValue':")
    print(df['MedHouseValue'].unique())

    # Checking for missing values in the dataset
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())


    """
    # Identify potential outliers or data quality issues using multiple methods
    # METHOD 1: Extreme Values (Percentiles)
    # METHOD 2: IQR (Interquartile Range) Method
    # METHOD 3: Zero or Negative Values
    # 
    # METHOD 4: Capped/Repeated Maximum Values
    # METHOD 5: Logical Inconsistencies (e.g., more bedrooms than rooms)
    # METHOD 6: Missing Values
    # 
    # Percentiles - Find values in extreme 1st/99th percentile
    IQR Method - Identifies points beyond 1.5×IQR from quartiles
    Statistical Bounds - Z-score or standard deviation method
    Domain Knowledge - Check for illogical values (negative prices, bedrooms > rooms)
    Capped Values - Repeated maximum values (data collection limit)
    Missing Data - NaN or null values
    Distribution Analysis - Skewness or unusual patterns in histograms
    """

    # METHOD 1: Check for extreme values using percentiles
    print("\n=== METHOD 1: Extreme Values (Percentiles) ===")
    print(df['MedHouseValue'].quantile([0.01, 0.25, 0.5, 0.75, 0.99]))
    # Values at 1st and 99th percentiles help identify extreme outliers

    # METHOD 2: Use IQR (Interquartile Range) method for outlier detection
    print("\n=== METHOD 2: IQR Method ===")
    Q1 = df['MedHouseValue'].quantile(0.25)
    Q3 = df['MedHouseValue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = df[(df['MedHouseValue'] < lower_bound) | (df['MedHouseValue'] > upper_bound)]
    print(f"Outliers detected: {len(iqr_outliers)}")
    print(iqr_outliers)

    # METHOD 3: Check for zero or negative values (data quality issue)
    print("\n=== METHOD 3: Zero or Negative Values ===")
    zero_negative = df[(df['MedInc'] <= 0) | (df['HouseAge'] < 0) | (df['Population'] <= 0)]
    print(f"Zero/Negative values found: {len(zero_negative)}")
    print(zero_negative)

    # METHOD 4: Check for capped values (suspicious repeated maximum values)
    print("\n=== METHOD 4: Capped/Repeated Maximum Values ===")
    capped = df[df['MedHouseValue'] == df['MedHouseValue'].max()]
    print(f"Rows with maximum value (5.0): {len(capped)}")
    print(capped.head())

    # METHOD 5: Check for unusual relationships (e.g., more bedrooms than rooms)
    print("\n=== METHOD 5: Logical Inconsistencies ===")
    logical_issues = df[df['AveBedrms'] > df['AveRooms']]
    print(f"Records where bedrooms > rooms: {len(logical_issues)}")
    print(logical_issues)

    # METHOD 6: Check for missing values and data types
    print("\n=== METHOD 6: Missing Values ===")
    print(df.isnull().sum())


    # Display statistical summary of the 'AveRooms' column (average number of rooms per household)
    print("\nStatistical summary of 'AveRooms':")
    print(df['AveRooms'].describe())

    # Display statistical summary of the 'Population' column (block population)
    print("\nStatistical summary of 'Population':")
    print(df['Population'].describe())

    # Display the top 5 most frequently occurring values in 'MedHouseValue' with their counts
    print("\nTop 5 most frequent values in 'MedHouseValue':")
    print(df['MedHouseValue'].value_counts().head())

    # -------------------------------
    # Boxplots for Outlier Detection
    # -------------------------------
    plt.figure()
    df.boxplot(column=['AveRooms', 'Population'])
    plt.title("Boxplots for Outlier Detection")
    plt.show()

    # -------------------------------
    # IQR-based Outlier Detection
    # -------------------------------
    ave_rooms_outliers = detect_outliers_iqr(df, 'AveRooms')
    population_outliers = detect_outliers_iqr(df, 'Population')

    print("\nNumber of AveRooms outliers (IQR):", len(ave_rooms_outliers))
    print("Number of Population outliers (IQR):", len(population_outliers))

    # -------------------------------
    # Z-score-based Outlier Detection
    # -------------------------------
    pop_z_outliers = detect_outliers_zscore(df, 'Population')
    print("Number of Population outliers (Z-score):", len(pop_z_outliers))

    # -------------------------------
    # Remove Outliers using IQR
    # -------------------------------
    df_cleaned = remove_outliers_iqr(df, 'AveRooms')
    df_cleaned = remove_outliers_iqr(df_cleaned, 'Population')

    print("\nShape before outlier removal:", df.shape)
    print("Shape after outlier removal:", df_cleaned.shape)

    # -------------------------------
    # Feature Scaling (Standardization)
    # -------------------------------
    features = df_cleaned.drop(columns=['MedHouseValue'])
    target = df_cleaned['MedHouseValue']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled['MedHouseValue'] = target.values

    print("\nScaled feature sample:")
    print(df_scaled.head())

   

if __name__ == "__main__":
    main()