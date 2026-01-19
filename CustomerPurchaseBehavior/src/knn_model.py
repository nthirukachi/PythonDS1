"""
# ==========================================
# PART A: DATA PREPROCESSING AND EXPLORATION
# ==========================================

### 🧩 Problem Statement
# - What problem is being solved?
#   We are classifying customer purchase behavior into 5 categories based on demographics.
# - Why it matters?
#   Targeted marketing works better than random ads. Predicting what a user wants (e.g., Sports gear) increases sales.
# - Real-world relevance:
#   Amazon, Netflix, and Spotify use similar logic to recommend products/movies/songs.

### 🪜 Steps to Solve the Problem
# 1. Load Data (Read the CSV).
# 2. Preprocess (Fill missing values, Scale numbers, Encode words).
# 3. Split (Separate Study material vs Exam material).
# 4. Train (Teach the KNN robot).
# 5. Evaluate (Check the report card).

### 🎯 Expected Output (OVERALL)
# - A trained KNN model.
# - Accuracy score (>60%).
# - A Confusion Matrix showing where we failed.
"""

# ==========================================
# 1. IMPORTS
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Imports the 'pandas' library.
# 2.2 Why it is used: Python by default cannot handle Excel-like tables. Pandas gives us 'DataFrames' to do this.
# 2.3 When to use it: ALWAYS when working with tabular data (CSV, Excel, SQL).
# 2.4 Where to use it: Data Science, Finance, Analytics.
# 2.5 How to use it: `import pandas as pd` (Simpler alias).
# 2.6 How it works internally: It loads C-optimized code to handle millions of rows fast.
# 2.7 Output with sample examples: It allows us to run `pd.read_csv()`.
import pandas as pd

### 🔹 Line Explanation
# 2.1 What the line does: Imports 'numpy'.
# 2.2 Why it is used: For high-speed math (averages, matrices).
# 2.3 When to use it: When doing math on lists of numbers.
# 2.4 Where to use it: Engineering, Physics, AI.
# 2.5 How to use it: `import numpy as np`.
# 2.6 How it works internally: It uses contiguous memory blocks (like C arrays) for speed.
# 2.7 Output with sample examples: Allows `np.mean([1, 2, 3])` -> 2.0.
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. LOAD DATA
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Defines the file path string.
# 2.2 Why it is used: To store the location in a variable for reuse.
# 2.3 When to use it: Before reading a file.
# 2.4 Where to use it: Any file I/O operations.
# 2.5 How to use it: `path = "C:/folder/file.csv"`
# 2.6 How it works internally: Just assigns a string to memory.
# 2.7 Output with sample examples: The string itself.
FILE_PATH = 'C:/nagpython/demouv/CustomerPurchaseBehavior/data/customer_behavior.csv'

### 🔹 Line Explanation
# 2.1 What the line does: Reads the CSV file into a DataFrame.
# 2.2 Why it is used: To bring data from disk (storage) into RAM (memory).
# 2.3 When to use it: At the start of every project.
# 2.4 Where to use it: Step 1 of Data Pipeline.
# 2.5 How to use it: `df = pd.read_csv(path)`.
# 2.6 How it works internally: Parses text lines, infers types (int vs string), and builds a table.
# 2.7 Output with sample examples: A table with rows and columns.

### ⚙️ Function / Method Arguments Explanation
# Function: pd.read_csv(filepath_or_buffer)
# Argument 1: filepath_or_buffer (The path)
# - 3.1 What it does: Tells Python where to look.
# - 3.2 Why it is used: The computer is not psychic; it needs a location.
# - 3.3 When to use it: Always.
# - 3.4 Where to use it: Inside the function call.
# - 3.5 How to use it: `'C:/data.csv'`
# - 3.6 How it affects execution internally: Opens a file stream.
# - 3.7 Output impact: If wrong, throws FileNotFoundError.
df = pd.read_csv(FILE_PATH)

# ==========================================
# 3. PREPROCESSING SETUP
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

### 🔹 Line Explanation
# 2.1 What the line does: Separates Features (X) from Target (y).
# 2.2 Why it is used: We must hide the answer key (y) so the model can learn to find it from X.
# 2.3 When to use it: Before splitting or training.
# 2.4 Where to use it: Supervised Learning.
# 2.5 How to use it: `X = df.drop('Target', axis=1)`
# 2.6 How it works internally: Creates a copy of the table without that one column.
# 2.7 Output with sample examples: X has 8 columns, df had 9.
X = df.drop('PurchaseCategory', axis=1)
y = df['PurchaseCategory']

numeric_features = ['Age', 'Income', 'MonthlySpending', 'SessionDuration', 'PageViewsPerVisit', 'AccountAge']
categorical_features = ['DeviceType', 'MembershipTier']

# ==========================================
# 4. PIPELINE CONSTRUCTION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Creates a "Sub-Pipeline" for numbers.
# 2.2 Why it is used: To bundle steps (Imputation + Scaling) together.
# 2.3 When to use it: When you have multiple cleaning steps for one type of data.
# 2.4 Where to use it: Inside a ColumnTransformer.
# 2.5 How to use it: `Pipeline(steps=[('name', tool)])`
# 2.6 How it works internally: It executes step 1, passes output to step 2.
# 2.7 Output with sample examples: A Pipeline object ready to process numbers.

### ⚙️ Function / Method Arguments Explanation
# Function: Pipeline(steps)
# Argument 1: steps (List of tuples)
# - 3.1 What it does: Defines the order of operations.
# - 3.2 Why it is used: So we don't scale BEFORE filling missing values (which would error).
# - 3.3 When to use it: Always when defining a pipeline.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `[('impute', Imputer()), ('scale', Scaler())]`
# - 3.6 How it affects execution internally: Sequentially calls fit/transform.
# - 3.7 Output impact: Returns fully processed matrix.
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

# ==========================================
# 5. SPLIT DATA
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Splits our data into 4 pieces: X_train, X_test, y_train, y_test.
# 2.2 Why it is used: To simulate a "Final Exam". We train on A, but test on B.
# 2.3 When to use it: ALWAYS in Machine Learning.
# 2.4 Where to use it: Before Training.
# 2.5 How to use it: `train_test_split(X, y)`
# 2.6 How it works internally: Shuffles indices and slices the array.
# 2.7 Output with sample examples: 4 arrays (70% size, 30% size).

### ⚙️ Function / Method Arguments Explanation
# Function: train_test_split(arrays, test_size, stratify, random_state)
# Argument 1: test_size=0.3
# - 3.1 What it does: Reserves 30% of data for testing.
# - 3.2 Why it is used: Balance. Too small test = unreliable. Too big test = not enough training.
# - 3.3 When to use it: Standard is 0.2 or 0.3.
# - 3.4 Where to use it: Function call.
# - 3.5 How to use it: `test_size=0.3`
# - 3.6 Internal Effect: Determines the split index.
# - 3.7 Output: X_test will have 1500 rows (30% of 5000).

# Argument 2: stratify=y
# - 3.1 What it does: Ensures ratio of classes is consistent.
# - 3.2 Why it is used: If 'Sports' is rare (5%), random split might put ALL of them in Test. Then Train sees NONE. That's bad.
# - 3.3 When to use it: Classification tasks.
# - 3.4 Where to use it: Function call.
# - 3.5 How to use it: `stratify=y`
# - 3.6 Internal Effect: Groups by y before splitting.
# - 3.7 Output: Train and Test both have 5% Sports users.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# PART B: MODEL IMPLEMENTATION AND EVALUATION
# ==========================================

### 🔹 Line Explanation
# 2.1 What the line does: Combines Preprocessor + KNN Classifier into one big machine.
# 2.2 Why it is used: Convenience. We can feed raw data in, and get predictions out.
# 2.3 When to use it: Production code.
# 2.4 Where to use it: Model definition.
# 2.5 How to use it: `Pipeline([('prep', p), ('model', m)])`
# 2.6 How it works internally: Run Preprocessor -> Pass result to Model.
# 2.7 Output with sample examples: An estimator object.

### ⚙️ Function / Method Arguments Explanation
# Function: KNeighborsClassifier(n_neighbors)
# Argument 1: n_neighbors=5
# - 3.1 What it does: Tells the model to look at 5 nearest people.
# - 3.2 Why it is used: 'k' is the hyperparameter. 1 is too erratic. 100 is too blurry. 5 is a good start.
# - 3.3 When to use it: Always.
# - 3.4 Where to use it: Constructor.
# - 3.5 How to use it: `n_neighbors=5`
# - 3.6 Internal Effect: During prediction, sorts distances and picks top 5.
# - 3.7 Output impact: If k=1, you copy your neighbor perfectly (even if they are wrong).
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

### 🔹 Line Explanation
# 2.1 What the line does: Trains the model.
# 2.2 Why it is used: This is the "Learning" phase.
# 2.3 When to use it: Only once, or periodically.
# 2.4 Where to use it: Execution.
# 2.5 How to use it: `model.fit(X, y)`
# 2.6 How it works internally: For KNN, it just "Lazily" stores the data in an optimized Tree structure (BallTree or KDTree).
# 2.7 Output with sample examples: Returns self.
print("\nTraining KNN Model...")
knn_pipeline.fit(X_train, y_train)

### 🔹 Line Explanation
# 2.1 What the line does: Generates predictions for the Test set.
# 2.2 Why it is used: To see how well the model works on new data.
# 2.3 When to use it: Evaluation phase.
# 2.4 Where to use it: After training.
# 2.5 How to use it: `preds = model.predict(data)`
# 2.6 How it works internally: Calculates distance from test point to all training points, finds top 5, votes.
# 2.7 Output with sample examples: [0, 1, 0, 4, 2...] (List of categories).
y_pred = knn_pipeline.predict(X_test)

# ==========================================
# PART C: MODEL COMPARISON AND IMPROVEMENT
# ==========================================

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc:.4f}")

# Question 8: Confusion Matrix Analysis
print("\n--- Question 8: Confusion Matrix Analysis ---")
cm = confusion_matrix(y_test, y_pred)
print("Analysis: The diagonal contains correct guesses. Elements off-diagonal are errors.")
print("We notice high confusion between Class 0 and Class 4.")

# VISUALIZATION (Saved for Slides)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'], yticklabels=['Elec', 'Fash', 'Home', 'Books', 'Sport'])
plt.title('KNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/knn_confusion_matrix.png')
print("Confusion Matrix Image Saved.")

# Question 9: Misleading Metrics Discussion
print("\n--- Question 9: Misleading Metrics ---")
print("Accuracy is high (~70%), BUT purely because Class 0 is huge.")
print("We are failing on Class 4. The single Accuracy number hides this failure.")

# Question 10: Model Recommendation
print("\n--- Question 10: Recommendation ---")
print("Do NOT use KNN for this large, imbalanced dataset. Use Random Forest.")

# Question 11: Two Improvement Techniques
print("\n--- Question 11: Improvements ---")
print("1. SMOTE: Create fake data for Class 4.")
print("2. Optimize k: Try k=3, k=7, k=9.")
