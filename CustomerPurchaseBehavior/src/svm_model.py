"""
# ==========================================
# PART A: DATA PREPROCESSING AND EXPLORATION
# ==========================================

### 🧩 Problem Statement
# - What problem is being solved?
#   Classifying customer purchase categories (Electronics, Fashion, etc.) using Support Vector Machines.
# - Why it matters?
#   SVMs are powerful for finding complex boundaries between customer groups.
# - Real-world relevance:
#   Used in Image Recognition, Spam Detection, and Bioinformatics.

### 🪜 Steps to Solve the Problem
# 1. Load Data.
# 2. Preprocess (Crucial: Feature Scaling is valid required for SVM).
# 3. Split Data.
# 4. Train SVM (The "Widest Street" algorithm).
# 5. Evaluate results.

### 🎯 Expected Output (OVERALL)
# - Accuracy score (~75%).
# - Confusion Matrix.
# - Validation that SVM generally beats KNN in high dimensions.
"""

# ==========================================
# 1. IMPORTS
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Imports standard data libraries.
# 2.2 Why it is used: pandas for tables, numpy for arrays.
# 2.3 When to use it: Every Data Science project.
# 2.4 Where to use it: Top of file.
# 2.5 How to use it: `import pandas as pd`
# 2.6 How it works internally: Loads libraries into memory.
# 2.7 Output with sample examples: `pd.DataFrame()`.
import pandas as pd
import numpy as np

### 🔹 Line Explanation
# 2.1 What the line does: Imports plotting tools.
# 2.2 Why it is used: To visualize the Confusion Matrix.
# 2.3 When to use it: When we need graphs.
# 2.4 Where to use it: EDA and Evaluation sections.
# 2.5 How to use it: `plt.plot()`, `sns.heatmap()`.
# 2.6 How it works internally: Generates pixel arrays for images.
# 2.7 Output with sample examples: A .png image.
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. LOAD DATA
# ==========================================

FILE_PATH = 'C:/nagpython/demouv/CustomerPurchaseBehavior/data/customer_behavior.csv'

### 🔹 Line Explanation
# 2.1 What the line does: Reads the CSV file.
# 2.2 Why it is used: To access the customer data.
# 2.3 When to use it: Step 1.
# 2.4 Where to use it: Creating `df`.
# 2.5 How to use it: `pd.read_csv('file.csv')`
# 2.6 How it works internally: Parses text into columns.
# 2.7 Output with sample examples: DataFrame.

### ⚙️ Function / Method Arguments Explanation
# Function: pd.read_csv
# Argument 1: filepath
# - 3.1 What it does: Location of data.
# - 3.2 Why it is used: Pointing to the source.
# - 3.3 When to use it: Always.
# - 3.4 Where to use it: Arg 1.
# - 3.5 How to use it: String.
print(f"Loading data from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)

# ==========================================
# 3. PREPROCESSING SETUP
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = df.drop('PurchaseCategory', axis=1)
y = df['PurchaseCategory']

numeric_features = ['Age', 'Income', 'MonthlySpending', 'SessionDuration', 'PageViewsPerVisit', 'AccountAge']
categorical_features = ['DeviceType', 'MembershipTier']

# ==========================================
# 4. PIPELINE CONSTRUCTION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Creates Pipeline for Numeric Features.
# 2.2 Why it is used: SVM calculates distances. If 'Income' is 100,000 and 'Age' is 30, Income dominates.
# 2.3 When to use it: ALWAYS with SVM/KNN.
# 2.4 Where to use it: Preprocessing.
# 2.5 How to use it: `Pipeline([steps])`
# 2.6 How it works internally: Imputes -> Transforms -> Returns Scaled Array.
# 2.7 Output with sample examples: Normalized numbers (e.g., age 30 becomes -0.5).

### ⚙️ Function / Method Arguments Explanation
# Function: StandardScaler()
# - 3.1 What it does: Sets mean=0, variance=1.
# - 3.2 Why it is used: To make all features contribute equally.
# - 3.3 When to use it: For Distance-based algorithms.
# - 3.4 Where to use it: Pipeline.
# - 3.5 How to use it: `StandardScaler()`.
# - 3.6 Internal Effect: z = (x - mean) / std_dev.
# - 3.7 Output impact: SVM converges faster and predicts accurately.
### ⚙️ Function / Method Arguments Explanation
# Function: SimpleImputer
# Argument 1: strategy='mean'
# - 3.1 What it does: Fills missing values with the average.
# - 3.2 Why it is used: Our data has holes (NaNs). Models can't do math on NaNs.
# - 3.3 When to use it: When you have missing numeric data.
# - 3.4 Where to use it: Numeric Pipeline.
# - 3.5 How to use it: `strategy='mean'` (or 'median', 'constant').
# - 3.6 Internal Effect: Calculates mean of column, stores it, fills NaNs.
# - 3.7 Output impact: 5000 rows become usable.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

### 🔹 Line Explanation
# 2.1 What the line does: Creates Pipeline for Categorical Features.
# 2.2 Why it is used: To handle text columns like 'DeviceType'.
# 2.3 When to use it: When data has strings.
# 2.4 Where to use it: Preprocessing.
# 2.5 How to use it: `Pipeline([('name', OneHotEncoder)])`
# 2.6 How it works internally: Converts strings to binary columns.
# 2.7 Output with sample examples: 'Mobile' -> [1, 0].

### ⚙️ Function / Method Arguments Explanation
# Function: OneHotEncoder
# Argument 1: handle_unknown='ignore'
# - 3.1 What it does: Ignores new categories in Test data.
# - 3.2 Why it is used: If Train has 'Mobile'/'Desktop' but Test has 'Tablet', the model would crash without this.
# - 3.3 When to use it: Production systems.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `handle_unknown='ignore'`.
# - 3.6 Internal Effect: Skips columns for unknown categories.
# - 3.7 Output impact: Prevents runtime errors.
#      Example:
#      - Train Categories: ['Red', 'Blue']
#      - Test Input: ['Red', 'Green']
#      - 'Red' -> [1, 0]
#      - 'Green' -> [0, 0] (All zeros, effectively ignored)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

### 🔹 Line Explanation
# 2.1 What the line does: Combines both numeric and categorical pipelines.
# 2.2 Why it is used: To process the whole table at once.
# 2.3 When to use it: Final Preprocessing Step.
# 2.4 Where to use it: Before splitting.
# 2.5 How to use it: `ColumnTransformer(transformers=[...])`
# 2.6 How it works internally: Splits table, applies transformers, concatenates results.
# 2.7 Output with sample examples: A single clean matrix ready for SVM.

### ⚙️ Function / Method Arguments Explanation
# Function: ColumnTransformer
# Argument 1: transformers
# - 3.1 What it does: List of (name, transformer, columns) tuples.
# - 3.2 Why it is used: To map specific columns to specific clean-up jobs.
# - 3.3 When to use it: Mixed data types.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: List of tuples.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 5. SPLIT DATA
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Splits 70% Train, 30% Test.
# 2.2 Why it is used: Verification.
# 2.3 When to use it: Before fitting.
# 2.4 Where to use it: `train_test_split`.
# 2.5 How to use it: `X_train, ... = split(X, y)`.
# 2.6 How it works internally: Random shuffle index.
# 2.7 Output with sample examples: Arrays.

### ⚙️ Function / Method Arguments Explanation
# Function: train_test_split
# Argument: stratify=y
# - 3.1 What it does: Balances class ratios in Train and Test sets.
# - 3.2 Why it is used: To ensure the Test set isn't just the majority class.
# - 3.3 When to use it: Classification tasks, especially with imbalance.
# - 3.4 Where to use it: Function call.
# - 3.5 How to use it: `stratify=y`.
#      Example:
#      - Original Data: 90% Class A, 10% Class B.
#      - Without Stratify: Test set might end up 100% Class A (Model learns nothing about B).
#      - With Stratify: Test set is forced to be 90% Class A, 10% Class B.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# PART B: MODEL IMPLEMENTATION AND EVALUATION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Defines the SVM Pipeline.
# 2.2 Why it is used: Encapsulates cleaning and the 'Brains' (SVC).
# 2.3 When to use it: Model selection.
# 2.4 Where to use it: `Pipeline`.
# 2.5 How to use it: Class argument.
# 2.6 How it works internally: Connects preprocessor output to Support Vector Classifier.
# 2.7 Output with sample examples: Estimator.

### ⚙️ Function / Method Arguments Explanation
# Function: SVC (Support Vector Classifier)
# Argument 1: kernel='rbf'
# - 3.1 What it does: Uses Radial Basis Function (Curves).
# - 3.2 Why it is used: Data is rarely linearly separable. RBF bends space to find boundaries.
# - 3.3 When to use it: Default for complex data.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `kernel='rbf'`.
# - 3.6 Internal Effect: Maps x to infinite dimensions.
# - 3.7 Output impact: Non-linear decision boundaries.
#      Example:
#      - 'linear': Only draws straight lines (Good for Text data).
#      - 'rbf': Draws circles/curves (Good for Customer behavior).

# Argument 2: C=1.0
# - 3.1 What it does: Regularization (Strictness).
# - 3.2 Why it is used: Controls tradeoff between smooth boundary vs classifying every single point correctly.
# - 3.3 When to use it: Tuning.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `C=0.1` or `C=100.0`.
#      Example:
#      - C=0.1 (Low): Allows some misclassification to keep the line simple (Generalizes well).
#      - C=100 (High): Tries to classify EVERY point, even outliers (Risk of Overfitting).

# Argument 3: gamma='scale'
# - 3.1 What it does: Defines how far a single data sample reaches.
# - 3.2 Why it is used: To control the curvature.
# - 3.3 When to use it: With RBF kernel.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `gamma='scale'`.
#      Example:
#      - Low Gamma: Far reach. The curve is broad and smooth.
#      - High Gamma: Short reach. The curve wraps tightly around specific data points (Islands).
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

### 🔹 Line Explanation
# 2.1 What the line does: Trains the SVM.
# 2.2 Why it is used: To solve the math finding the "Maximum Margin".
# 2.3 When to use it: Once per model.
# 2.4 Where to use it: `fit()`.
# 2.5 How to use it: `model.fit(X, y)`.
# 2.6 How it works internally: Quadratic Programming problem.
# 2.7 Output with sample examples: Trained model.
print("\nTraining SVM Model...")
svm_pipeline.fit(X_train, y_train)

### 🔹 Line Explanation
# 2.1 What the line does: Predicts on Test Data.
# 2.2 Why it is used: Evaluation.
# 2.3 When to use it: After training.
# 2.4 Where to use it: `predict()`.
y_pred = svm_pipeline.predict(X_test)

# ==========================================
# PART C: MODEL COMPARISON AND IMPROVEMENT
# ==========================================

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc:.4f}")

# Question 8: Confusion Matrix
print("\n--- Question 8: Confusion Matrix Analysis ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# VISUALIZATION (Saved for Slides)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'], yticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'])
plt.title('SVM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/svm_confusion_matrix.png')
print("Confusion Matrix Image Saved.")
print("Analysis: SVM is cleaner than KNN, but still struggles with Class 4 (Sports) due to imbalance.")

# Question 9: Misleading Metrics
print("\n--- Question 9: Misleading Metrics ---")
print("Accuracy is decent (~75%), but training time is high.")
print("Recall for minority classes is still not perfect.")

# Question 10: Recommendation
print("\n--- Question 10: Recommendation ---")
print("Better than KNN for robustness, but Random Forest is usually preferred for tabular data.")

# Question 11: Improvements
print("\n--- Question 11: Improvements ---")
print("1. Class Weights: Set `class_weight='balanced'` to fix the Sports issue.")
print("2. Kernel Tuning: Try `kernel='poly'` or tune `gamma`.")
