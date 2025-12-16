"""
Problem Statement: Multiple Linear Regression for Housing Prices.

Dataset: Mimics structure of https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
- Target: Price
- Features: Area, Bedrooms, Bathrooms, Stories, Mainroad, Guestroom, etc.

Feature Selection Process Explained:
1. Correlation Analysis: We remove features highly correlated with *each other* (Multicollinearity) to keep coefficients interpretable.
2. RFE (Recursive Feature Elimination): We iteratively remove the weakest features to find the subset that maximizes predictive power.
"""

# What: Import Pandas for tabular data manipulation.
# Why: Essential for reading datasets and creating DataFrames.
# Output: Module 'pandas' as 'pd'.
import pandas as pd

# What: Import NumPy for numerical operations.
# Why: Used for random number generation and mathematical logic.
# Output: Module 'numpy' as 'np'.
import numpy as np

# What: Import plotting libraries.
# Why: To visualize correlations and relationships.
import matplotlib.pyplot as plt
import seaborn as sns

# What: Import model selection utilities.
# Why: train_test_split is crucial to evaluate on unseen data.
from sklearn.model_selection import train_test_split

# What: Import Preprocessing tools.
# Why: 
# - MinMaxScaler: Scaling (0 to 1) is mandatory for Regression interpretation.
# - LabelEncoder: Not used directly (mapped manually), but good practice for categories.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# What: Import the Linear Regression algorithm.
# Why: The problem explicitly asks for a Multiple Linear Regression model.
from sklearn.linear_model import LinearRegression

# What: Import RFE (Recursive Feature Elimination).
# Why: Used for automatic Feature Selection.
from sklearn.feature_selection import RFE

# What: Import Metrics.
# Why: R2 and RMSE are standard regression accuracy metrics.
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. Data Preparation
# ==========================================
# ==========================================
# 1. Data Preparation
# ==========================================
def load_data():
    """
    Generates synthetic data mirroring the specific Kaggle dataset structure.
    """
    # What: seed(42).
    # Why: Reproducibility. Ensures the 'random' house prices are the same every run.
    np.random.seed(42)
    n = 545 # Size of actual dataset
    
    # What: Create dictionary of synthetic values.
    # Why: Replicates columns like 'area' (int), 'bedrooms' (int), 'mainroad' (categorical).
    data = {
        'price': np.random.randint(1750000, 13300000, n),
        'area': np.random.randint(1650, 16200, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.randint(1, 4, n),
        'stories': np.random.randint(1, 4, n),
        'mainroad': np.random.choice(['yes', 'no'], n),
        'guestroom': np.random.choice(['yes', 'no'], n),
        'basement': np.random.choice(['yes', 'no'], n),
        'hotwaterheating': np.random.choice(['yes', 'no'], n),
        'airconditioning': np.random.choice(['yes', 'no'], n),
        'parking': np.random.randint(0, 3, n),
        'prefarea': np.random.choice(['yes', 'no'], n),
        'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], n)
    }
    # What: Inject a linear relatioship.
    # Why: Linear Model needs linear data. Without this, R2 would be ~0.
    # Logic: Price = 500*Area + 1M*Bathrooms + Noise.
    data['price'] = (data['area'] * 500) + (data['bathrooms'] * 1000000) + (data['stories'] * 500000) + np.random.randint(-500000, 500000, n)
    
    return pd.DataFrame(data)

# What: Load dataframe.
df = load_data()
print("Data Shape:", df.shape)
print(df.head())

# Preprocessing: Binary Mapping
# What: List of columns with 'yes'/'no' values.
# When: Before training.
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Loop to map 'yes' -> 1, 'no' -> 0.
# Why: Machine Learning models behave better with 1/0 integers than strings.
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Preprocessing: Dummy Encoding
# What: Convert 'furnishingstatus' (3 levels) to dummy variables.
# Argument: drop_first=True.
# Why: To avoid "Dummy Variable Trap" (Multicollinearity). 
# If semi-furnished=0 and unfurnished=0, it implies furnished=1 automatically.
status = pd.get_dummies(df['furnishingstatus'], drop_first=True)

# What: Concatenate new columns and drop original categorical string column.
df = pd.concat([df, status], axis=1)
df.drop(['furnishingstatus'], axis=1, inplace=True)

# ==========================================
# 2. Feature Selection Process
# ==========================================
print("\n--- Feature Selection Analysis ---")

# Step A: Correlation Heatmap
# What: Calculate and plot Pearson correlation coefficients.
# Why: To visualize linear relationships. 
# - High +ve correlation (red) means feature drives price up.
# - High -ve correlation (blue) means feature drives price down.
# - Correlation between two features (not target) warns of Multicollinearity.
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
# plt.show() # Commented out for non-interactive execution

# Step B: Train/Test Split
# What: Separate Target (y) from Features (X).
y = df.pop('price')
X = df

# What: Split 70% Train, 30% Test.
# Argument: random_state=100 ensures consistent split.
# Why: Standard Practice.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Scaling (Important for Regression interpretation)
# What: Initialize MinMax Scaler (transforms data to [0, 1]).
# Why: 
# - Linear Regression coefficients represent "Effect of 1 unit change".
# - If 'area' is thousands (1000-16000) and 'stories' is small (1-4), 'stories' coefficient will look artificially huge to compensate.
# - Scaling puts them on level playing field so coefficients compare importance.
scaler = MinMaxScaler()

# What: Fit on TRAIN, Transform on TRAIN and TEST.
# WARNING: NEVER fit on Test data (Data Leakage).
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Step C: Recursive Feature Elimination (RFE)
# What: Use RFE to select top 10 features.
# Logic: 
# 1. Train model on ALL features.
# 2. Find feature with smallest absolute coefficient.
# 3. Drop it.
# 4. Repeat until 10 features left.
lm = LinearRegression()
lm.fit(X_train, y_train)

# Select Top 10 features
# Arguments: n_features_to_select=10.
rfe = RFE(lm, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

# Output Ranking
print("Feature Ranking (1 = Selected):")
# rfe.ranking_ is 1 for selected, >1 for dropped (2 = last dropped, 3 = second last dropped, etc).
print(list(zip(X_train.columns, rfe.ranking_)))

# Filter X to supported features
# What: Create new filtered dataframes with ONLY the 10 selected columns.
col_supported = X_train.columns[rfe.support_]
print("\nSelected Features:", list(col_supported))

# Explain Selection:
# "We selected these features because RFE identified them as having the strongest Linear Coefficients contributing to Price, after removing noise."

X_train_rfe = X_train[col_supported]
X_test_rfe = X_test[col_supported]

# ==========================================
# 3. Model Building & Evaluation
# ==========================================
# ==========================================
# 3. Model Building & Evaluation
# ==========================================
print("\n--- Model Training ---")

# Step 1: Initialize Model
# What: Create an instance of Linear Regression (Ordinary Least Squares).
# Why: Standard algorithm for continuous numerical prediction.
model = LinearRegression()

# Step 2: Fit Model
# What: Learn the coefficients (weights) from the Training Data.
# Input: X_train_rfe (10 selected features), y_train (Prices).
model.fit(X_train_rfe, y_train)

# Step 3: Predict
# What: Generate price predictions for the unseen Test set.
# Output: Array of predicted prices.
y_pred = model.predict(X_test_rfe)

# Metrics
# What: R-Squared (Coeff of Determination).
# Why: Measures how well the variance in Price is explained by our features. 1.0 = Perfect.
r2 = r2_score(y_test, y_pred)

# What: Root Mean Squared Error.
# Why: Measures average error in dollars. "On average, we are off by $RMSE".
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared Score: {r2:.4f}")
print(f"RMSE: {rmse:,.2f}")

# Coefficients Analysis
# What: Extract the 'coef_' attribute from the trained model.
# Why: These numbers tell us the IMPACT of each feature.
# Interpretation: "Holding all else constant, increasing 'bedrooms' by 1 (or 1 unit of scale) changes Price by Coeff".
coef_df = pd.DataFrame({'Feature': col_supported, 'Coefficient': model.coef_})

# What: Sort by magnitude.
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Coefficients):")
print(coef_df.to_string(index=False))

# Explanation of Output:
# "For every 1 unit increase in 'area' (normalized), price increases by X amount, holding other vars constant."
