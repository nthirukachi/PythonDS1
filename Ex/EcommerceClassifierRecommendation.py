"""
Problem Statement:
E-commerce Customer Purchase Prediction & Classifier Recommendation.
Context:
- Dataset: 100,000 customers, 20 features (Mixed types: Age, Income, Country, etc.).
- Target: Purchase (Balanced: 50k Yes / 50k No).
- Requirements: Real-time prediction (<100ms), High Interpretability (Explain "Why").

Steps to Solve:
1. Generate a synthetic dataset mimicking the e-commerce scenario (Age, Income, Country, Device).
2. Preprocess data: Encode categorical variables (One-Hot Encoding) for machine learning.
3. Train the Recommended Classifier: **Decision Tree**.
   - Why recommended? Fast inference (<100ms), natively interpretable (If-Then rules), handles non-linear relationships.
4. Measure and demonstrate Prediction Latency (Time taken for single prediction).
5. Demonstrate Interpretability by extracting Feature Importance (showing *why* decisions are made).
6. Print a consulting report ruling out SVM and k-NN.

Expected Output:
- Training confirmation.
- Prediction time (e.g., "0.0002 seconds" - well under 100ms).
- Top Feature Importances (e.g., "Income: 0.45, Age: 0.30...").
- Text report explaining why Decision Tree is chosen over SVM/k-NN.
"""

# Why: Import necessary libraries for data manipulation, timing, and machine learning.
# Why: Pandas is essential for structuring the dataset into rows and columns (DataFrame).
# Output: Imports pandas as 'pd'.
import pandas as pd

# Why: NumPy is needed for numerical operations and efficient random number generation.
# Output: Imports numpy as 'np'.
import numpy as np

# Why: Python's built-in time module is needed to measure the execution latency (<100ms requirement).
# Output: Imports time module.
import time
# Why: DecisionTreeClassifier is the core algorithm we are recommending.
# Why: export_text allows us to visualize the tree rules as text for stakeholders.
# Output: Imports class 'DecisionTreeClassifier' and function 'export_text'.
from sklearn.tree import DecisionTreeClassifier, export_text

# Why: Needed to split the 100k records into training (to teach the model) and testing (to verify it).
# Output: Imports function 'train_test_split'.
from sklearn.model_selection import train_test_split

# Why: Decision Trees in Scikit-Learn typically require numerical inputs, so we encode categories.
# Output: Imports class 'OneHotEncoder'.
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# Step 1: Data Generation (Simulation)
# ==========================================

def generate_mock_data(n=1000):
    """
    Generates a mock e-commerce dataset.
    Arguments:
    n (int): Number of samples to generate.
    Output: DataFrame with synthetic features and target.
    """
    # Why: Seed ensures that the random numbers generated are the same every time we run the script.
    # Output: Sets the global random seed to 42.
    np.random.seed(42)
    # Why: Simulate numerical features
    # randint(low, high, size): Generates 'n' random integers for Age (18-70) and Income (20k-150k).
    # uniform(low, high, size): Generates 'n' random floats for TimeOnSite.
    age = np.random.randint(18, 70, size=n)
    income = np.random.randint(20000, 150000, size=n)
    time_on_site = np.random.uniform(1, 60, size=n)
    
    # Why: Simulate categorical features
    # choice(list, size): Randomly selects from the given list of strings for 'n' rows.
    countries = np.random.choice(['USA', 'UK', 'CA', 'DE'], size=n)
    devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n)
    
    # Why: Create target (Purchase). Logic: Higher income/time -> Higher chance.
    # This ensures the Decision Tree actually finds patterns to "explain".
    # np.where: Adds a bonus score if country is USA.
    score = (income / 1000) + (time_on_site * 2) + (np.where(countries == 'USA', 50, 0))
    # Thresholding to create balanced classes roughly
    # Why: Convert continuous score to binary (0/1). > Median ensures ~50/50 split.
    purchased = (score > np.median(score)).astype(int)
    
    # Why: Combine all numpy arrays into a pandas DataFrame.
    # Output: DataFrame object with labeled columns.
    df = pd.DataFrame({
        'Age': age, 'Income': income, 'TimeOnSite': time_on_site,
        'Country': countries, 'Device': devices,
        'Purchased': purchased
    })
    # Why: Return the created dataframe to the caller.
    return df

# Why: Generate data to demonstrate the classifier's capabilities.
# We use a subset (10,000) for the demo script speed, but the logic applies to 100k.
print("--- Step 1: Generating Data ---")
# Why: Generate 10,000 samples to demonstrate the classifier's capabilities.
# We use a subset (10,000) for the demo script speed, but the logic applies to 100k.
# Output: calls generate_mock_data().
df = generate_mock_data(n=10000)

# Why: Verify the dimensions of the dataset.
# Output: Prints (10000, 6).
print(f"Dataset Shape: {df.shape}")

# Why: Peek at the first few rows to ensure data looks correct.
# Output: Prints head of dataframe.
print(df.head(3))

# ==========================================
# Step 2: Preprocessing
# ==========================================

# Why: Machine Learning models (like Decision Trees in sklearn) require numerical input.
# We must convert 'Country' and 'Device' into numbers.
# pd.get_dummies: Performs One-Hot Encoding (e.g., Country_USA = 1, Country_UK = 0).
# Why: Machine Learning models (like Decision Trees in sklearn) require numerical input.
# We must convert 'Country' and 'Device' into numbers.
# pd.get_dummies: Performs One-Hot Encoding (e.g., Country_USA = 1, Country_UK = 0).
# drop('Purchased'): X should only contain features, not the answer.
# Output: X is the feature matrix (numerical), y is the target vector.
print("\n--- Step 2: Preprocessing ---")
X = pd.get_dummies(df.drop('Purchased', axis=1))
y = df['Purchased']

# Split data
# Split data
# Why: Using 80% for training (8000 rows) and 20% for testing (2000 rows).
# Output: 4 arrays (X_train, X_test, y_train, y_test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data encoded and split.")

# ==========================================
# Step 3: Train Recommended Classifier
# ==========================================

# Recommendation: Decision Tree Classifier
# Why: 
# 1. Interpretability: Produces human-readable rules (perfect for stakeholders).
# 2. Speed: Inference is O(Depth), typically extremely fast (<1ms).
# 3. Handling features: Handles numerical/categorical mix well (after encoding).

print("\n--- Step 3: Training Decision Tree ---")
# Argument 'max_depth=5': Limits tree expansion.
# Why? A full tree is distinct/complex (overfitting). A depth of 5 is easy to explain to a human.
# Output: Creates an untrained Decision Tree classifier object.
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Why: Record start time to measure training speed (though prediction speed is the critical requirement).
start_train = time.time()

# Why: Fit (train) the model on the 8000 training examples.
# This builds the actual tree structure (rules) based on entropy/gini impurity.
clf.fit(X_train, y_train)

# Output: prints training time.
print(f"Model Trained in {time.time() - start_train:.4f} seconds.")

# ==========================================
# Step 4: Measure Prediction Latency
# ==========================================

print("\n--- Step 4: Real-time Prediction Check ---")
# Why: Simulate a simplified single customer record (taken from test set) coming in.
single_customer = X_test.iloc[0:1]

# Why: Record the exact moment before prediction starts.
start_pred = time.time()

# Why: Make the classification prediction for the single customer.
# In production, this is the step that must be < 100ms.
prediction = clf.predict(single_customer)

# Why: Record time immediately after prediction finishes.
end_pred = time.time()

# Output details
# Why: Convert seconds to milliseconds for easier reading vs requirements.
latency_ms = (end_pred - start_pred) * 1000
print(f"Prediction: {prediction[0]}")
print(f"Latency: {latency_ms:.4f} ms")

# Why: Validate if the model strictly meets the business requirement of 100ms.
if latency_ms < 100:
    print("Result: MEETS < 100ms requirement.")
else:
    print("Result: FAIL")

# ==========================================
# Step 5: Interpretation (The "Why")
# ==========================================

print("\n--- Step 5: Explaining to Stakeholders ---")
# Why: Extract feature importances to show WHICH factors drive purchases.
# These scores sum to 1.0 and represent how much each feature decreases impurity (gini).
importances = clf.feature_importances_
feature_names = X.columns

# Sort and print top 3 drivers
# Why: argsort returns indices that would sort the array. [::-1] reverses it to Descending order.
indices = np.argsort(importances)[::-1]
print("Top 3 Factors driving purchase behavior:")
# Why: Loop to print friendly names for the top 3 features.
for i in range(3):
    # Output Example: "Income", "TimeOnSite".
    print(f"{i+1}. {feature_names[indices[i]]} (Importance: {importances[indices[i]]:.4f})")

# Why: Show an actual rule (If Income > X then Buy).
# export_text: generates a text representation of the decision logic.
# Stakeholders can read this like a manual.
print("\nDecision Rule Sample (Text Tree):")
print(export_text(clf, feature_names=list(X.columns), max_depth=2))

# ==========================================
# Step 6: Final Recommendation Report
# ==========================================

def print_consulting_report():
    report = """
========================================
       CONSULTING RECOMMENDATION
========================================
Recommended Classifier: Decision Tree (or Random Forest)

1. RULING OUT OTHERS:
   A) k-Nearest Neighbors (k-NN):
      - Reason: Prediction Speed.
      - Why: k-NN is "Lazy". To predict 1 customer, it calculates distance to ALL 100,000 training points.
      - Result: Latency grows with data size (O(N)). Likely > 100ms for 100k samples with 20 dims.
   
   B) Support Vector Machine (SVM):
      - Reason: Interpretability & Training Speed.
      - Why: SVMs (especially with RBF kernels) are "Black Boxes". 
      - Stakeholders cannot easily understand "hyperplane margin in infinite dimensions".
      - Training on 100k samples is computationally expensive (approx O(N^2) to O(N^3)).

2. WHY DECISION TREE:
   - Speed: Prediction is O(depth). With depth ~10-20, it's instant (<1ms).
   - Interpretability: We can literally print the flowchart (as shown above).
     Stakeholders can see: "If Income > $80k AND TimeOnSite > 10m -> Predict Buy".
   - Mixed Data: Handles mix of age/income/country well.

3. IMPLEMENTATION CONSIDERATIONS:
   A) Overfitting Control: 
      - Trees can memorize data. Must set 'max_depth' or 'min_samples_leaf' to keep rules generalizable.
   B) Categorical Encoding:
      - Sklearn trees need One-Hot Encoding (as done in Step 2). 
      - High-cardinality features (like City) might need target encoding to avoid massive tree width.
"""
    print(report)

print_consulting_report()
