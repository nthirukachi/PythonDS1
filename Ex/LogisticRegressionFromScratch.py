"""
Problem Statement:
Implementing Logistic Regression from Scratch.
Dataset: Breast Cancer Wisconsin (Diagnostic).
Goal: Predict tumor malignancy (Malignant/Benign).

Requirements:
1. Load & Preprocess Data (Split 80/20, Standardize).
2. Implement from scratch:
   - Sigmoid Function
   - Binary Cross-Entropy Cost
   - Gradient Descent Optimization
3. Compare with Scikit-Learn.
4. Visualize Cost vs Iterations.

Expected Output:
- Cost convergence plot.
- Test Accuracy of Scratch Model vs Sklearn Model.
- Detailed logic explaining the math behind logistic regression.
"""

# Why: Import NumPy for matrix operations (Dot product, Exponentials).
# Output: Module 'numpy' loaded as 'np'.
import numpy as np

# Why: Import Matplotlib to visualize the training loss curve.
# Output: Module 'pyplot' loaded.
import matplotlib.pyplot as plt

# Why: Import Scikit-Learn tools for Data Loading, splitting, and benchmarking.
# Output: Functions loaded.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Class Definition: Logistic Regression
# ==========================================

# ==========================================
# 2. Functional Implementation (No Class)
# ==========================================

# sigmoid:
# What: Activation function S(z) = 1 / (1 + e^-z).
# When: Used in prediction to map linear output (z) to probability (0 to 1).
# Why: Fundamental to Logistic Regression; differentiable and bounded.
# Arguments: z (Linear combination Xw + b).
# Output: Probability value(s) between 0 and 1.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# train_logistic_regression:
# What: Trains the model using Gradient Descent.
# When: Called on training data.
# Steps:
# 1. Initialize weights/bias to zero.
# 2. Loop n_iterations:
#    a. Forward pass (compute probability).
#    b. Compute Gradient (Derivative of Loss w.r.t weights).
#    c. Update parameters (weights = weights - learning_rate * gradient).
#    d. Calculate and store loss (Cost).
# Arguments: 
# - X: Features matrix
# - y: Target vector
# - learning_rate: Step size
# - n_iterations: Loop count
# Output: Tuple (weights, bias, cost_history)
def train_logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    # Initialize parameters
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    cost_history = []

    # Gradient Descent Loop
    for i in range(n_iterations):
        # 1. Linear Model (z = wX + b)
        # Output: Linear score vector.
        linear_model = np.dot(X, weights) + bias
        
        # 2. Activation (Predictions)
        # Output: Probability vector y_pred.
        y_predicted = sigmoid(linear_model)

        # 3. Compute Gradients (Derivatives)
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # 4. Update Parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # 5. Compute Cost (Binary Cross Entropy)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
        cost = - (1 / n_samples) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        cost_history.append(cost)

        # Optional: Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")
            
    return weights, bias, cost_history

# predict:
# What: Generates class labels (0 or 1).
# When: Evaluation phase.
# Why: We need hard classifications for accuracy, not just probabilities.
# Logic: If prob > 0.5 -> 1 (Malignant), else 0 (Benign).
# Arguments: 
# - X: Features
# - weights: Trained weights
# - bias: Trained bias
# Output: Array of binary labels.
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

# ==========================================
# 2. Main Execution Block
# ==========================================

print("--- 1. Data Loading & Preprocessing ---")

# Load Data
# What: Fetches the Wisconsin Breast Cancer Dataset.
# Why: Standard benchmark for binary classification.
# Output (data): Dictionary-like object with 'data' and 'target'.
data = load_breast_cancer()
X = data.data
y = data.target

print(f"Features: {X.shape}")
print(f"Target Distribution: {np.bincount(y)} (0: Malignant, 1: Benign)") 
# Note: In sklearn dataset, 0=Malignant, 1=Benign usually, or vice versa depending on version. 
# We treat it as generic binary 0/1.

# Split Data
# What: 80% Train, 20% Test.
# Output (X_train): Training features.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
# What: Scale features to mean=0, std=1.
# Why: CRITICAL for Gradient Descent. If features have vastly different ranges (e.g., 0.01 vs 1000), 
# the gradients will oscillate and converge slowly or diverge.
# Output (scaler): Trained scaler object.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- 2. Training Implementation from Scratch ---")

# Initialize Model
# Arguments: Learning Rate=0.01, Iterations=1000.
# Why: Standard starting points.
# model_scratch = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)

# Fit Model
# What: Runs the gradient descent loop.
# model_scratch.fit(X_train, y_train)
weights, bias, cost_history = train_logistic_regression(X_train, y_train, learning_rate=0.1, n_iterations=1000)

# Predict
# What: Get predictions on test set.
# y_pred_scratch = model_scratch.predict(X_test)
y_pred_scratch = predict(X_test, weights, bias)

# Calculate Accuracy
acc_scratch = accuracy_score(y_test, y_pred_scratch)
print(f"\nScratch Model Accuracy: {acc_scratch:.4f}")

print("\n--- 3. Comparison with Scikit-Learn ---")

# Train Sklearn Model
# What: Use the optimized library version.
# Why: To benchmark our implementation.
clf_sklearn = LogisticRegression(random_state=42)
clf_sklearn.fit(X_train, y_train)
y_pred_sklearn = clf_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn Model Accuracy: {acc_sklearn:.4f}")
print(f"Defference: {abs(acc_scratch - acc_sklearn):.4f}")

# ==========================================
# 3. Visualization
# ==========================================

print("\n--- 4. Visualizing Cost History ---")

# Plotting
# What: Draws the Loss vs Iteration curve.
# Why: To verify that Gradient Descent actually converged (Loss should decrease).
# Output: A pop-up window or saved image of the plot.
plt.figure(figsize=(10, 6))
# plt.plot(range(len(model_scratch.cost_history)), model_scratch.cost_history, color='blue')
plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.title('Cost Function vs Iterations (Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Cost (Binary Cross Entropy)')
plt.grid(True)
plt.show()
print("Plot displayed successfully.")
