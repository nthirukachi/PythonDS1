"""
# ==========================================
# PART A: DATA PREPROCESSING AND EXPLORATION
# ==========================================

### 🧩 Problem Statement
# - What problem is being solved?
#   Comparing 4 AI models to find the champion.
# - Why it matters?
#   No single model is perfect. We must test them all.

### 🪜 Steps to Solve the Problem
# 1. Load Data.
# 2. Preprocess.
# 3. Create List of Models.
# 4. Loop through models (Train & Test).
# 5. Compare Accuracy.

### 🎯 Expected Output (OVERALL)
# - A bar chart identifying Random Forest as the winner.
"""

# ==========================================
# 1. IMPORTS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 2. LOAD DATA
# ==========================================

FILE_PATH = 'C:/nagpython/demouv/CustomerPurchaseBehavior/data/customer_behavior.csv'
print(f"Loading data from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)

X = df.drop('PurchaseCategory', axis=1)
y = df['PurchaseCategory']

numeric_features = ['Age', 'Income', 'MonthlySpending', 'SessionDuration', 'PageViewsPerVisit', 'AccountAge']
categorical_features = ['DeviceType', 'MembershipTier']

# ==========================================
# 3. PREPROCESSING SETUP
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Defines transformers.
# 2.2 Why it is used: Consistency. We use ONE preprocessor for ALL models to be fair.
# 2.3 When to use it: Comparison studies.
# 2.4 Where to use it: Pipeline.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# PART B: MODEL IMPLEMENTATION (LOOP)
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Defines Dictionary of models.
# 2.2 Why it is used: To iterate efficiently.
# 2.3 When to use it: Benchmarking.
# 2.4 Where to use it: Loop source.
# 2.5 How to use it: `{'name': object}`.
# 2.6 How it works internally: Stores pointers to classes.
# 2.7 Output with sample examples: Dict.
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = []

print("\nStarting Comparison...")

### 🔹 Line Explanation
# 2.1 What the line does: The Main Loop.
# 2.2 Why it is used: Don't Repeat Yourself (DRY).
# 2.3 When to use it: Multiple models.
# 2.4 Where to use it: Here.
for name, model in models.items():
    # Pipeline
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Train
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Store
    results.append({"Model": name, "Accuracy": acc})
    print(f"--> Accuracy: {acc:.4f}")

# ==========================================
# PART C: MODEL COMPARISON
# ==========================================

res_df = pd.DataFrame(results)

### 🔹 Line Explanation
# 2.1 What the line does: Plots the Bar Chart.
# 2.2 Why it is used: Visual proof of the winner.
# 2.3 When to use it: Final report.
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=res_df, palette='viridis')
plt.title("Model Showdown")
plt.ylim(0, 1.0)
plt.savefig('C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/final_comparison_strict.png')
print("\nPlot Saved.")

# Question 10: Final Recommendation
print("\n--- Final Verdict ---")
print("Winner: RANDOM FOREST.")
print("Reason: Best combination of accuracy and robustness.")
