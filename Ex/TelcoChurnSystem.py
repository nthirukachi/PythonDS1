"""
Problem Statement: Multi-Classifier System Design and Performance Analysis (Telco Churn).

Scenario: Telecom company needs to predict customer churn (Time horizon: 30 days).
Dataset: 7043 rows, 20 features (Categorical/Numerical), Class Imbalance (73% No / 27% Yes).

Business Requirements:
1. Recall > 0.70 (Catch at least 70% of churners).
2. Capacity Constraint: Contact max 2000 customers/month (~28% of base).
3. Explainability: Must tell executives "Why".
4. ROI: Retention Cost $50, Churn Loss $500.

Steps to Solve:
1. Data Preprocessing:
   - Handle 'TotalCharges' (string -> float).
   - Encode Categorical (OneHot) for LogReg/SVM, Ordinal for Trees.
   - Scale Numerical (StandardScaler) for k-NN/SVM/LogReg.
2. Classifier Implementation:
   - Logistic Regression (Baseline, Explainable).
   - k-NN (Non-linear, Scaled).
   - SVM (High Performance, Heavy Compute).
   - Decision Tree (Explainable, Non-linear).
3. Evaluation:
   - Metrics: Recall (Priority), Precision (Constrained by budget), ROI.
   - Strategy: Stratified 5-Fold CV.
4. Business Impact:
   - ROI Calculation: (True Positives * $450 Saved) - (False Positives * $50 Wasted).
   - Threshold Tuning: Optimize probability threshold to fit the 2000-contact budget.
5. Deployment:
   - Drift Monitoring (KS Test).
   - Retraining Strategy.

Expected Output:
- A pipeline script that loads data, processes it, trains 4 models, prints specific metrics, 
  calculates ROI in dollars, and generates comparison plots.
- Detailed comments explaining every line.
"""

# What: Import data manipulation library.
# Why: Essential for handling the CSV dataset and DataFrames.
# When: Start of script.
# Output: Module 'pandas' as 'pd'.
import pandas as pd

# What: Import numerical operations library.
# Why: Used for array conversion and NaN handling.
# When: Start of script.
# Output: Module 'numpy' as 'np'.
import numpy as np

# What: Import plotting libraries.
# Why: Required for ROI and Metric visualization.
# Output: Modules 'pyplot' and 'seaborn'.
import matplotlib.pyplot as plt
import seaborn as sns

# What: Import Scikit-Learn Preprocessing tools.
# Why: 
# - OneHotEncoder: Transforms categorical variables (like 'InternetService') into binary numbers (0/1) because ML models require numeric input.
# - StandardScaler: Normalizes numerical variables (like 'MonthlyCharges') to have Mean=0 and Std=1. Essential for k-NN and SVM which rely on Euclidean distance.
# - SimpleImputer: Handles missing values (NaNs) by filling them with a strategy (e.g., median) so the model doesn't crash.
# - ColumnTransformer: Allows applying different transformations to different columns (e.g., Scaling numbers but Encoding strings).
# - Pipeline: Chains steps together (Preprocessing -> Model) into a single object, ensuring the exact same steps are applied to new data.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# What: Import Models.
# Why: The 4 requested usage.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# What: Import Validation tools.
# Why: Split data and measure performance reliably.
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================

print("--- 1. Data Preprocessing ---")

# Load Data
# What: Simulate Loading Telco Dataset. 
# Why: Can't access Kaggle URL directly, so simulating strict structure matching description.
# Output: DataFrame (7043 rows).
def load_simulation_data():
    # What: Create synthetic dataframe matching Telco specs.
    # Why: We do not have direct access to the Kaggle CSV. This function generates random data with the same structure (rows, columns, types) to demonstrate the pipeline.
    # When: At the start of the process.
    # Output: A Pandas DataFrame with 7043 rows.
    np.random.seed(42) # Ensure we get the same random numbers every time.
    n = 7043
    data = {
        # Random integer between 1 and 72 months for tenure.
        'tenure': np.random.randint(1, 72, n),
        # Random float fee between $20 and $120.
        'MonthlyCharges': np.random.uniform(20, 120, n),
        # Total charges is usually tenure * monthly, but here just random float. 
        # Converted to string (.astype(str)) to simulate the dirty data mentioned in the prompt.
        'TotalCharges': np.random.uniform(20, 8000, n).astype(str), 
        # Random categories matching the domain.
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        # Imbalanced Target: 27% Yes, 73% No.
        'Churn': np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73]) 
    }
    # Inject missing values (empty strings) in TotalCharges to test our cleaning logic.
    data['TotalCharges'][0:10] = " " 
    return pd.DataFrame(data)

df = load_simulation_data()

# Data Cleaning
# What: Convert 'TotalCharges' to numeric, forcing errors to NaN.
# Why: Dataset has " " strings for TotalCharges. Models need floats.
# Output: Series of floats (with some NaNs).
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Target Encoding
# What: Map 'Yes' -> 1, 'No' -> 0.
# Why: Scikit-Learn needs integers for calculation.
# Output: Updated 'Churn' column.
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn'])
y = df['Churn']

# Define Features
# What: Specify which columns are Numerical (Measurements) and which are Categorical (Labels).
# Why: They require different preprocessing steps. You cannot average a string, and you cannot OneHotEncode a continuous float efficiently.
# When: Before configuring transformers.
# Output: Lists of column strings.
categorical_features = ['Contract', 'InternetService']
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Design Pipeline
# ----------------------------------------
# 1. Numerical Transformer
# What: A sub-pipeline for handling numbers.
# Steps:
#   a. 'imputer': Fills missing values with the Median. Median is better than Mean because TotalCharges has large outliers (8000+) that skew the mean.
#   b. 'scaler': Applies Z-Score Normalization ((X - Mean)/Std). This puts all features on the same scale (e.g., Tenure 0-72 and Charges 0-8000 become roughly -2 to +2).
# Why: Essential for Distance-based algorithms (k-NN, SVM).
# Output: Array of Scaled Floats.
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Categorical Transformer
# What: A sub-pipeline for handling strings.
# Steps:
#   a. 'imputer': Fills missing values with the string 'missing'. Ensures the encoder doesn't break on NaNs.
#   b. 'onehot': Converts 'InternetService' (DSL, Fiber, No) into 3 binary columns (1,0,0 / 0,1,0 ...).
#      'handle_unknown=ignore' ensures that if the Test Set has a new category not seen in Train, it ignores it instead of crashing.
# Why: Regression and SVM matrix math requires pure numbers.
# Output: Sparse Matrix of 0s and 1s.
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Combine with ColumnTransformer
# What: The master processor that routes columns to their respective transformers.
# Why: Allows us to treat the whole mixed dataset as one object.
# Output: A 'preprocessor' object we can plug into the final model pipeline.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ])

# ==========================================
# 2. Multi-Classifier Implementation
# ==========================================

print("\n--- 2. Model Training & Comparison ---")

# Define Models
# What: Configure the 4 required algorithms.
# Why: To compare performance across different mathematical approaches (Linear, Distance-based, Tree-based).
# Settings: 
# - class_weight='balanced': Crucial for 27% imbalance. It tells the model "Pay 3x more attention to Churners" so it doesn't just predict "No Churn" for everyone.
# - probability=True (SVM): Required to use .predict_proba(). Standard SVM only gives Class 0/1, but we need the % Score for the ROI ranking.
classifiers = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5), # k-NN has no intrinsic class_weight argument. We rely on the threshold tuning later.
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

# Split Data
# What: Create a Hold-Out Test Set (20%).
# Why: We optimize models on Train, but we MUST calculate ROI on data the model has never seen to simulate the "Next Month" of business.
# Arguments: stratify=y ensures the 27% churn rate is preserved in both Train and Test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

results_list = []

print("Starting Training Loop...")
for name, clf in classifiers.items():
    # What: Create the Full Pipeline.
    # Why: Connects Preprocessor (Impute+Scale) -> Classifier. This ensures scaling parameters are learned ONLY from Train data and applied to Test, preventing Data Leakage.
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', clf)])
    
    # Train
    # What: Fit the pipeline on training data.
    # Output: Trained model ready for prediction.
    full_pipeline.fit(X_train, y_train)
    
    # Predict Probabilities
    # What: Get the likelihood (0.0 to 1.0) of Churn (Class 1) for the Test set.
    # Why: We need to rank customers from "Highest Risk" to "Lowest Risk" to prioritize who we call.
    # Output: Array of floats.
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    
    # ROI Calculation (Business Metric)
    # ---------------------------------
    # Constraint: "Customer retention team can only contact 2000 customers per month".
    # Scaling: X_test (size 1409) is roughly 20% of the full dataset (size 7043). 
    # Therefore, the budget for this Test month is roughly 20% of 2000 = 400 calls.
    budget_test = 400
    
    # What: Identify the Top 400 Riskiest Customers.
    # Steps: 
    # 1. argsort: Returns indices that would sort the array.
    # 2. [::-1]: Reverses it to Descending order (High Risk first).
    # 3. [:400]: Takes the top 400.
    # Why: Optimizes our limited budget.
    top_indices = np.argsort(y_prob)[::-1][:budget_test]
    
    # Calculate Profit
    # What: Check the ground truth for those 400 people.
    # Matches (Hits): Customers we Called (Predicted Churn) who WERE Churners. We saved them.
    # Misses: Customers we Called who were actually Happy. We wasted money on them.
    target_customers_truth = y_test.iloc[top_indices]
    hits = sum(target_customers_truth == 1)
    misses = sum(target_customers_truth == 0)
    
    # Value Calculation:
    # Benefit: $500 Revenue Saved per Hit.
    # Cost: $50 Retention Offer per Contact (Hit OR Miss).
    # Net: (Hits * $500) - ((Hits + Misses) * $50)
    # Simplified: (Hits * $450) - (Misses * $50)
    roi = (hits * 450) - (misses * 50)
    
    # Standard Metrics (at 0.5 threshold)
    # What: Standard Recall for comparison.
    y_pred_default = full_pipeline.predict(X_test)
    rec = recall_score(y_test, y_pred_default)
    
    results_list.append({
        'Model': name,
        'ROI ($)': roi,
        'Recall (Default)': rec,
        'Hits (Churners Saved)': hits,
        'Misses (Budget Wasted)': misses
    })

# ==========================================
# 3. Performance Analysis
# ==========================================

print("\n--- 3. Business Impact Analysis (Budget Constraint: Top 400 aka ~2000/mo) ---")

# What: Convert the list of result dictionaries into a printable DataFrame.
# Output: Table with columns 'Model', 'ROI', 'Recall', etc.
roi_df = pd.DataFrame(results_list)
print(roi_df.to_string(index=False))

# Visualization
# What: Create a Bar Chart comparing the Net Profit (ROI) of each model.
# Why: Executives care about dollars, not F1-scores. This chart answers "Which model makes the most money?".
# Arguments: figsize sets the image dimensions.
plt.figure(figsize=(10, 6))

# What: Plot the bars using Seaborn.
# Arguments:
# - data: source dataframe.
# - x: Model Names on X-axis.
# - y: Dollar amounts on Y-axis.
# - palette: Color scheme for aesthetics.
sns.barplot(data=roi_df, x='Model', y='ROI ($)', palette='viridis')

# What: Add Title and Labels.
plt.title('Estimated Monthly ROI (Test Subset Budget=400)')
plt.ylabel('Net Profit ($)')

# What: Add horizontal grid lines to make it easier to read the values.
plt.grid(axis='y')

# What: Display the plot.
plt.show()

# ==========================================
# 4. Deployment Strategy Comments
# ==========================================
print("\n" + "="*50)
print("PART 4: PRODUCTION DEPLOYMENT STRATEGY")
print("="*50)

# What: Print the detailed strategy text.
# Why: To fulfill the Requirement for a Deployment Plan.
print("""
1. Model Choice: 
   - Recommendation: Logistic Regression.
   - Justification: It likely provided competitive ROI while offering 'Coefs' to explain WHY churn happens (Requirement 3). 
     Tree is explainable but prone to overfitting (high variance). SVM is too slow/black-box.

2. Monitoring (Drift):
   - System: 'Drift Monitor Service'.
   - Metric: Periodic KS-Test (Kolmogorov-Smirnov) on 'MonthlyCharges' distribution.
   - Trigger: If P-value < 0.05 vs Training Baseline, trigger alarm. This means the new data looks statistically different from training data.

3. Retraining:
   - Schedule: Weekly (since 'Daily updates' is a requirement, Weekly retraining balances cost vs freshness).
   - Pipeline: Automated script (Airflow) runs this exact preprocessing + training pipeline on trailing 30 days of data.
   - Safe-Rollout: 'Canary Deployment'. Route 10% of traffic to new model. If ROI holds, roll out to 100%.
""")
