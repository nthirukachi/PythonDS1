"""
Problem Statement:
Healthcare Diagnostic Tool Development: 3-Class Classification (Type A, Type B, Healthy).
constraints: 
- 10,000 Patient Records, 15 Continuous Features.
- Cost Sensitive: False Negative is 10x worse than False Positive.
- Mobile Deployment: Limited computational resources.
- Regulatory Requirement: Explainable predictions.

Steps to Solve:
1. Algorithm Selection: Evaluate k-NN, SVM, and Decision Tree. Select Decision Tree.
2. Data Simulation: Generate a synthetic dataset matching class distributions (3k, 2.5k, 4.5k).
3. Implementation Strategy: 
   - Training a Decision Tree with `class_weight` to handle the 10:1 cost ratio.
   - Limiting tree depth for mobile efficiency.
4. Validation: 
   - Metrics: Focus on Recall (Sensitivity) and Weighted F2-Score (which prioritizes Recall).
   - Mobile Test: Measure model size (KB) and inference latency (ms).
   - Interpretability: Visualize the decision rules.

Expected Output:
- Algorithm Comparison Matrix.
- Training performance metrics emphasizing Recall working on the minority/disease classes.
- Validation that the model is small (< few KB) and fast (<1ms) for mobile.
- Text representation of the decision logic for doctors.
"""

# Why: Import Pandas for tabular data handling and pretty printing the comparison matrix.
# Output: Module 'pandas' loaded.
import pandas as pd

# Why: Import NumPy for efficient numerical data generation.
# Output: Module 'numpy' loaded.
import numpy as np

# Why: Import Time to measure inference latency for the mobile constraint.
# Output: Module 'time' loaded.
import time

# Why: Import Pickle to measure the serialized model size (storage constraint).
# Output: Module 'pickle' loaded.
import pickle

# Why: Import Decision Tree (Chosen Algorithm) and metrics.
# Output: Classification classes and metric functions loaded.
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer

# ==========================================
# Phase 1: Algorithm Selection Matrix
# ==========================================

def print_algorithm_selection():
    """
    Prints a comparison matrix effectively performing the 'Select Algorithm' task.
    """
    # Why: Define the scores for each algorithm against requirements.
    # Scores: 1 (Poor), 2 (Medium), 3 (Excellent).
    comparison_data = {
        'Requirement': ['Accuracy Potential', 'Interpretability', 'Comp. Efficiency (Inference)', 'Class Imbalance Handling', 'Mobile Deployment'],
        'k-NN': ['High (3)', 'Low (1) - "Black Box"', 'Low (1) - O(N) Slow', 'Medium (2)', 'Low (1) - Stores all data'],
        'SVM': ['High (3)', 'Low (1) - Hard to explain', 'Medium (2)', 'Medium (2)', 'Medium (2)'],
        'Decision Tree': ['Medium/High (2)', 'High (3) - Explicit Rules', 'High (3) - O(Depth) Fast', 'High (3) - Class Weights', 'High (3) - Lightweight Rules']
    }
    
    # Why: specific weighting logic for choice.
    # Decision Tree wins on Interpretability + Mobile Deployment + Cost Handling.
    df_comp = pd.DataFrame(comparison_data)
    print("\n--- 1. Algorithm Selection Matrix ---")
    print(df_comp.to_string(index=False))
    print("\nSelected Algorithm: Decision Tree")
    print("Justification: Best balance of Explainability (Medical Reqt) and Efficiency (Mobile Reqt).")

print_algorithm_selection()

# ==========================================
# Phase 2: Data Simulation
# ==========================================

# Why: Generate synthetic medical data to simulate the environment.
# Output: X (features) and y (labels).
print("\n--- 2. Data Simulation ---")
np.random.seed(42)
n_patients = 10000

# Why: Generate 15 continuous blood markers.
# Output: Matrix of 10k x 15.
X = np.random.normal(loc=100, scale=20, size=(n_patients, 15))

# Why: Create labels based on distribution: 30% Type A (0), 25% Type B (1), 45% Healthy (2).
# We artificially inject a pattern so the model learns something.
# - If Feature 0 > 120 -> Likely Type A
# - If Feature 1 < 80  -> Likely Type B
# - Else Healthy
y = np.zeros(n_patients)
# Output: Assigning classes based on probability logic (simplified).
for i in range(n_patients):
    rand = np.random.rand()
    if rand < 0.3:
        y[i] = 0 # Type A
        # Shift marker 0 for Type A
        X[i, 0] += 30 
    elif rand < 0.55:
        y[i] = 1 # Type B
        # Shift marker 1 for Type B
        X[i, 1] -= 30
    else:
        y[i] = 2 # Healthy

# train_test_split: 
# What: Splits arrays or matrices into random train and test subsets.
# When: Before training the model to create a withheld dataset for honest validation.
# Why: Essential to evaluate how the model performs on unseen data (generalization).
# Arguments:
# - test_size=0.2: 20% of data (2000 patients) for validation.
# - stratify=y: Maintains the % of Type A/B diseases in train/test splits. IMPORTANT for imbalance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data Generated. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ==========================================
# Phase 3: Implementation Strategy (Training)
# ==========================================

print("\n--- 3. Implementation Strategy ---")

# Strategy: Handling 10:1 Cost Ratio
# Why: Standard accuracy treats all errors equally. In healthcare, missing a disease (FN) is disastrous.
# We define class weights to penalize errors on Disease (0, 1) more than Healthy (2).
# Weight 10 for Disease, 1 for Healthy.
# Output: Dictionary mapping class index to weight power.
cost_weights = {0: 10, 1: 10, 2: 1}

# Strategy: Mobile Deployment Optimization
# Why: Limit max_depth to keep the model small (KB) and fast (few if-checks).
# Output: Hyperparameters set.
mobile_depth_limit = 10

# Why: Initialize the Decision Tree with these strategies.
# Output: Model object ready for training.
clf = DecisionTreeClassifier(
    # criterion='gini': 
    # What: The mathematical formula used to calculate "impurity" (how mixed the classes are in a node).
    # When: Used at every split decision during training.
    # Why: 'gini' is faster to compute than 'entropy' and usually yields similar results.
    criterion='gini',
    
    # class_weight=cost_weights: 
    # Value: {0: 10, 1: 10, 2: 1}.
    # What: Assigns a penalty multiplier to each class. Misclassifying class 0 costs 10x more than class 2.
    # When: Used during the calculation of pureness/error to decide where to split.
    # Why: Critical for "10x worse" requirement. It forces the model to prioritize minimizing False Negatives on Disease.
    class_weight=cost_weights, 
    
    # max_depth=10:
    # What: The maximum number of levels (questions) the tree is allowed to grow down.
    # When: Used as a stopping condition during the recursive tree building process.
    # Why: 
    # 1. Mobile Constraint: A deeper tree (e.g., 50) takes more RAM and CPU. Depth 10 is lightweight.
    # 2. Overfitting: Prevents the model from memorizing noise in the 10k records, ensuring better generalization.
    max_depth=mobile_depth_limit, 
    
    # random_state=42:
    # What: The seed value for the internal random number generator.
    # When: Used whenever the algorithm needs to make a random choice (e.g., picking best split among ties).
    # Why: Ensures the tree structure is identical every time we run the script (Reproducibility).
    random_state=42
)

# clf.fit(X, y):
# What: The core training function. Builds the decision tree.
# When: After model initialization and before prediction.
# Why: It recursively finds the best features/thresholds to minimize Gini Impurity (weighted by class_weight).
clf.fit(X_train, y_train)
print("Model Trained with Cost-Sensitive Learning.")

# ==========================================
# Phase 4: Validation Plan
# ==========================================

print("\n--- 4. Validation Plan ---")

# clf.predict(X):
# What: Generates class labels for input samples.
# When: During evaluation or live deployment.
# Why: To verify if the learned rules generalize to the test set (X_test).
y_pred = clf.predict(X_test)

# Metric 1: Recall & Confusion Matrix
# Why: In high-risk diagnostics, Recall (Sensitivity) of the Disease classes is the #1 metric.
# We check if we are catching the Type A and B cases.
print("Confusion Matrix:")
# Output: Matrix [[TP_A, E, E], [E, TP_B, E], [E, E, TP_H]]
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Focus on Recall '0' and '1'):")
# Output: Precision/Recall table.
print(classification_report(y_test, y_pred, target_names=['Type A', 'Type B', 'Healthy']))

# Metric 2: Mobile Viability Check
# Why: Check if the model fits in limited memory.
# pickle.dumps: Serializes the model object to bytes.
# Output: Size in Kilobytes.
model_size = len(pickle.dumps(clf)) / 1024
print(f"Model Size: {model_size:.2f} KB (Excellent for Mobile)")

# Why: Check inference speed.
# Output: Average time per patient in milliseconds.
start_time = time.time()
# Simulate batch of 100 predictions
clf.predict(X_test[:100]) 
inference_per_sample = (time.time() - start_time) / 100 * 1000 # to ms
print(f"Inference Latency: {inference_per_sample:.4f} ms/patient (<< 100ms)")

# ==========================================
# Phase 5: Interpretability (Regulatory)
# ==========================================

print("\n--- 5. Regulatory Explainability ---")

# Why: Doctors need to know "Why" a patient is diagnosed Type A.
# Decision Tree allows us to print the exact Threshold Rules.
# Output: Text-based tree structure.
print("Diagnostic Rules (Snippet):")
feature_names = [f"Marker_{i}" for i in range(15)]
# export_text:
# What: Converts the internal tree structure into a string representation.
# When: Generating reports for stakeholders/regulators.
# Why: Satisfies the "Explainable AI" requirement by showing exactly WHY a decision was made.
print(export_text(clf, feature_names=feature_names, max_depth=2))

print("\nSummary:")
print("1. Strategy: Used Class Weights (10:1) to prioritize detecting diseases.")
print("2. Mobile: Limited depth resulted in small model size and fast latency.")
print("3. Explainability: Rules extracted above satisfy regulatory transparency.")
