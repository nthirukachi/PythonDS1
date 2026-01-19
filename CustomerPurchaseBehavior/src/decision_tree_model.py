"""
# ==========================================
# PART A: DATA PREPROCESSING AND EXPLORATION
# ==========================================

### 🧩 Problem Statement
# - What problem is being solved?
#   Building a flowchart-like model to classify customers.
# - Why it matters?
#   It's transparent. We can show the "Rules" to a business manager.
# - Real-world relevance:
#   RPA (Robotic Process Automation), Medical Triage Rules.

### 🪜 Steps to Solve the Problem
# 1. Load Data.
# 2. Preprocess (Impute missing data).
# 3. Split.
# 4. Train Decision Tree.
# 5. Visualize the Tree.

### 🎯 Expected Output (OVERALL)
# - Accuracy ~60-70%.
# - A graphical tree showing the decision logic.
"""

# ==========================================
# 1. IMPORTS
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Imports Pandas.
# 2.2 Why it is used: Data manipulation.
# 2.3 When to use it: Always.
# 2.4 Where to use it: Top.
# 2.5 How to use it: `import pandas as pd`
# 2.6 How it works internally: Loads libraries.
# 2.7 Output with sample examples: `pd`.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. LOAD DATA
# ==========================================

FILE_PATH = 'C:/nagpython/demouv/CustomerPurchaseBehavior/data/customer_behavior.csv'
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X = df.drop('PurchaseCategory', axis=1)
y = df['PurchaseCategory']

numeric_features = ['Age', 'Income', 'MonthlySpending', 'SessionDuration', 'PageViewsPerVisit', 'AccountAge']
categorical_features = ['DeviceType', 'MembershipTier']

# ==========================================
# 4. PIPELINE CONSTRUCTION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Preprocessing for numbers.
# 2.2 Why it is used: To fix missing values. Scaling is optional for Trees but kept for consistency.
# 2.3 When to use it: Preprocessing.
# 2.4 Where to use it: Pipeline.
# 2.5 How to use it: `Pipeline([...])`
# 2.6 How it works internally: Series of transformers.
# 2.7 Output with sample examples: Clean matrix.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#      Example:
#      - Train Categories: ['Red', 'Blue']
#      - Test Input: ['Red', 'Green']
#      - 'Red' -> [1, 0]
#      - 'Green' -> [0, 0] (All zeros, effectively ignored)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 5. SPLIT DATA
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Splits data.
# 2.2 Why it is used: Training vs Testing.
# 2.3 When to use it: Before fitting.
# 2.4 Where to use it: `train_test_split`.
# 2.5 How to use it: `train_test_split(X, y)`.
# 2.6 How it works internally: Shuffle and slice.
# 2.7 Output with sample examples: 4 arrays.
# - 3.5 How to use it: `stratify=y`.
#      Example:
#      - Original Data: 90% Class A, 10% Class B.
#      - Without Stratify: Test set might end up 100% Class A.
#      - With Stratify: Test set is forced to be 90% Class A, 10% Class B.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# PART B: MODEL IMPLEMENTATION AND EVALUATION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Defines Decision Tree Pipeline.
# 2.2 Why it is used: Creates the learner.
# 2.3 When to use it: Model definition.
# 2.4 Where to use it: `Pipeline`.
# 2.5 How to use it: Class argument.
# 2.6 How it works internally: Recursively splits data.
# 2.7 Output with sample examples: Estimator.

### ⚙️ Function / Method Arguments Explanation
# Function: DecisionTreeClassifier
# Argument 1: max_depth=4
# - 3.1 What it does: Limits tree height.
# - 3.2 Why it is used: To prevent overfitting (memorizing).
# - 3.3 When to use it: Always tune this.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `int`.
# - 3.6 Internal Effect: Stops split recursion at depth 4.
# - 3.7 Output impact: Simpler, more general model.
#      Example:
#      - max_depth=None: Tree takes every single student and memorizes their name.
#      - max_depth=2: Tree groups students by "Height" and "Age" only.

# Argument 2: class_weight='balanced'
# - 3.1 What it does: Weights rare classes higher.
# - 3.2 Why it is used: Imbalance handling.
# - 3.3 When to use it: When data is skewed.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `class_weight='balanced'`.
# - 3.6 Internal Effect: Multiplies loss function.
#      Example:
#      - If "Sports" is 10x rarer than "Electronics",
#      - Every mistake on "Sports" carries 10x penalty.
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42))
])

### 🔹 Line Explanation
# 2.1 What the line does: Trains the Tree.
# 2.2 Why it is used: Finds the best questions to ask.
# 2.3 When to use it: Once.
# 2.4 Where to use it: `fit()`.
# 2.5 How to use it: `model.fit(X, y)`.
# 2.6 How it works internally: CART algorithm (Gini Impurity).
# 2.7 Output with sample examples: Trained tree.
print("\nTraining Decision Tree...")
dt_pipeline.fit(X_train, y_train)

# Visualize
print("Plotting Tree...")
plt.figure(figsize=(20,10))
plot_tree(dt_pipeline.named_steps['classifier'], filled=True)
plt.savefig('C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/dt_strict_refactored.png')

### 🔹 Line Explanation
# 2.1 What the line does: Predicts.
# 2.2 Why it is used: Evaluation.
# 2.3 When to use it: After training.
# 2.4 Where to use it: `predict()`.
y_pred = dt_pipeline.predict(X_test)

# ==========================================
# PART C: MODEL COMPARISON AND IMPROVEMENT
# ==========================================

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc:.4f}")

# Question 8: Confusion Matrix
print("\n--- Question 8: Confusion Matrix Analysis ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Analysis: Using 'balanced' weights means we find more Class 4 (Sports) customers, but we might have more False Positives.")

# VISUALIZATION (Saved for Slides)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'], yticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'])
plt.title('Decision Tree Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/dt_confusion_matrix.png')
print("Confusion Matrix Image Saved.")

# Question 9: Misleading Metrics
print("\n--- Question 9: Misleading Metrics ---")
print("Accuracy is lower than SVM, but utility is higher because we explain the logic.")

# Question 10: Recommendation
print("\n--- Question 10: Recommendation ---")
print("Use this for reporting to humans. Use Random Forest for automated predictions.")

# Question 11: Improvements
print("\n--- Question 11: Improvements ---")
print("1. Pruning: Cut small branches.")
print("2. Ensemble: Use Random Forest.")
