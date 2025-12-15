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
    # ============================================================================
    # FEATURE SCALING & ENCODING
    # ============================================================================
    """
    WHY FEATURE SCALING IS IMPORTANT:
    
    1. **Standardization (StandardScaler)**:
       - Transforms features to have mean=0 and standard deviation=1
       - Formula: (X - mean) / std_dev
       - Important for algorithms sensitive to feature magnitude:
         * Distance-based algorithms (KNN, K-means)
         * Gradient descent-based algorithms (Linear/Logistic Regression, Neural Networks)
         * Regularized models (Ridge, Lasso, Elastic Net)
       - Without scaling: Features with larger ranges dominate the model
    
    2. **Why Categorical Encoding is Important**:
       - Machine learning algorithms work with numerical data only
       - OneHotEncoder converts categorical variables to binary dummy variables
       - Prevents artificial ordering in categorical features
       - Example: Color (Red=0, Blue=1, Green=2) would imply ordering
    
    3. **Benefits of Feature Scaling**:
       - Faster convergence in gradient descent optimization
       - Better generalization across different feature scales
       - Improved numerical stability in matrix operations
       - More interpretable coefficients in linear models
    """
    
    print("\n" + "="*80)
    print("FEATURE SCALING AND ENCODING")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Separate features (X) and target (y)
    # We drop the target column from features because models need them separately
    # =========================================================================
    # Remove target column 'MedHouseValue' from features to get independent variables
    features = df_cleaned.drop(columns=['MedHouseValue'])
    # Extract target column separately for model training (dependent variable)
    target = df_cleaned['MedHouseValue']
    
    # =========================================================================
    # STEP 2: Identify and categorize data types
    # This helps us apply appropriate preprocessing to each data type
    # =========================================================================
    # Select only numerical columns (int, float) to identify which need scaling
    # We use select_dtypes() to automatically detect column data types
    numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    # Select only categorical/text columns for OneHotEncoding
    # These need to be converted to numerical format for ML algorithms
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_columns)}): {numerical_columns}")
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    
    # =========================================================================
    # STEP 3: Apply StandardScaler to numerical features
    # StandardScaler transforms data: (X - mean) / std_dev → mean=0, std=1
    # This is crucial for distance-based and gradient descent algorithms
    # =========================================================================
    print("\n--- APPLYING STANDARDSCALER ---")
    # Display statistics before scaling to show the transformation
    print("Before scaling - Sample statistics:")
    print(f"MedInc: Mean={features['MedInc'].mean():.4f}, Std={features['MedInc'].std():.4f}")
    print(f"HouseAge: Mean={features['HouseAge'].mean():.4f}, Std={features['HouseAge'].std():.4f}")
    print(f"Population: Mean={features['Population'].mean():.4f}, Std={features['Population'].std():.4f}")
    
    # Create a StandardScaler object to standardize all numerical features
    # We use StandardScaler because it handles outliers better than MinMaxScaler
    scaler = StandardScaler()
    # fit_transform() learns the mean and std from training data, then transforms it
    # This prevents data leakage (scaling parameters derived from training data only)
    scaled_features_array = scaler.fit_transform(features[numerical_columns])
    
    # Convert the scaled numpy array back to DataFrame for easier manipulation
    # We preserve column names and index to maintain data alignment
    df_scaled_numerical = pd.DataFrame(
        scaled_features_array,  # The scaled numerical data
        columns=numerical_columns,  # Original column names for reference
        index=features.index  # Maintain row indices for proper alignment
    )
    
    # Display statistics after scaling to verify the transformation worked
    print("\nAfter scaling - Sample statistics:")
    print(f"MedInc: Mean={df_scaled_numerical['MedInc'].mean():.6f}, Std={df_scaled_numerical['MedInc'].std():.6f}")
    print(f"HouseAge: Mean={df_scaled_numerical['HouseAge'].mean():.6f}, Std={df_scaled_numerical['HouseAge'].std():.6f}")
    print(f"Population: Mean={df_scaled_numerical['Population'].mean():.6f}, Std={df_scaled_numerical['Population'].std():.6f}")
    
    # =========================================================================
    # STEP 4: Apply OneHotEncoder to categorical features (if any exist)
    # OneHotEncoder converts categorical variables into binary dummy variables
    # This is essential because ML algorithms require numerical input
    # =========================================================================
    if len(categorical_columns) > 0:
        # Import OneHotEncoder only if categorical columns exist (for efficiency)
        from sklearn.preprocessing import OneHotEncoder
        print("\n--- APPLYING ONEHOTENCODER ---")
        # Create encoder with sparse_output=False to get dense array (easier to work with)
        # handle_unknown='ignore' prevents errors if new categories appear in test data
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # fit_transform() learns categories from training data, then converts to binary columns
        # For example: Color 'Red' → [1,0,0], 'Blue' → [0,1,0], 'Green' → [0,0,1]
        encoded_categorical = encoder.fit_transform(features[categorical_columns])
        # get_feature_names_out() returns meaningful names like 'Color_Red', 'Color_Blue'
        # This makes the encoded features interpretable in the final dataset
        categorical_feature_names = encoder.get_feature_names_out(categorical_columns)
        # Convert encoded array to DataFrame for consistency with scaled numerical data
        # We maintain the same index to ensure proper alignment when combining
        df_encoded_categorical = pd.DataFrame(
            encoded_categorical,  # The encoded binary data
            columns=categorical_feature_names,  # Meaningful column names
            index=features.index  # Match the index of original features
        )
        # Combine scaled numerical features with encoded categorical features horizontally (axis=1)
        # axis=1 means we add new columns (not new rows)
        df_final = pd.concat([df_scaled_numerical, df_encoded_categorical], axis=1)
        print(f"Categorical features encoded: {len(categorical_feature_names)} new columns created")
    else:
        # If no categorical columns exist, use only the scaled numerical features
        df_final = df_scaled_numerical
        print("\n--- NO CATEGORICAL FEATURES FOUND ---")
    
    # =========================================================================
    # STEP 5: Add the target variable back to the final dataset
    # We keep features and target separate until here for proper preprocessing
    # =========================================================================
    # Add the target column 'MedHouseValue' back to the processed features
    # We use .values to convert Series to numpy array for proper alignment
    df_final['MedHouseValue'] = target.values
    
    # =========================================================================
    # STEP 6: Display and verify the final scaled dataset
    # =========================================================================
    print("\n--- FINAL SCALED DATASET ---")
    # Show the shape to understand dataset dimensions
    print(f"Shape: {df_final.shape}")
    # Display first few rows to visually inspect scaled values
    print("\nFirst 5 rows of scaled data:")
    print(df_final.head())
    
    # Show data types to confirm all columns are numerical (ready for ML)
    print("\nData types after scaling and encoding:")
    print(df_final.dtypes)
    
    # =========================================================================
    # STEP 7: Verify that scaling was successful
    # All numerical features should have mean ≈ 0 and standard deviation ≈ 1
    # =========================================================================
    print("\n--- SCALING VERIFICATION ---")
    print("All numerical features now have mean ≈ 0 and std ≈ 1")
    # Calculate mean of each scaled numerical feature (should be ≈ 0)
    print(f"Mean of scaled features:\n{df_final[numerical_columns].mean()}")
    # Calculate standard deviation (should be ≈ 1)
    print(f"\nStandard deviation of scaled features:\n{df_final[numerical_columns].std()}")

   

if __name__ == "__main__":
    main()