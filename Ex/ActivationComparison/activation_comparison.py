"""
====================================================================================================
1. PROBLEM STATEMENT:
Compare sigmoid versus ReLU activations in a shallow neural network on make_moons data.

We need to:
1.  Generate a synthetic "moons" dataset (non-linear).
2.  Preprocess the data (Standardization).
3.  Train two separate MLP Classifiers: one with Sigmoid activation, one with ReLU.
4.  Record and plot their Loss Curves to compare convergence speed.
5.  Evaluate their final accuracy and confusion matrices.
6.  Analyze how the choice of activation function impacts training dynamics.

STEPS TO SOLVE THE PROBLEM:
1.  Data Setup: Generate 800 samples, noise=0.25. Split 70/30.
2.  Preprocessing: Scale features using StandardScaler (crucial for Neural Nets).
3.  Model Definition:
    -   Model 1: MLP with `activation='logistic'` (Sigmoid).
    -   Model 2: MLP with `activation='relu'`.
    -   Both use hidden layers (20, 20) and max_iter=300.
4.  Training: Fit both models to training data.
5.  Evaluation:
    -   Extract `loss_curve_` from each model.
    -   Predict on test set.
    -   Compute Accuracy and Confusion Matrix.
6.  Visualization: Plot the two loss curves on the same graph/axes.

EXPECTED OUTPUT:
-   Console output showing accuracy for both models.
-   A plot showing ReLU converging significantly faster (steeper drop) than Sigmoid.
-   Confusion matrices for both models.
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports numpy (numerical python).
# 2.2: Why it is used: For array manipulations if needed.
# 2.3: When to used: Math operations.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `import numpy as np`.
# 2.6: Output: Library loaded.
import numpy as np

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports plotting library.
# 2.2: Why it is used: To visualize the loss curves.
# 2.3: When to used: Visualization.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `import matplotlib.pyplot as plt`.
# 2.6: Output: Library loaded.
import matplotlib.pyplot as plt

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports dataset generator.
# 2.2: Why it is used: To create the 'moons' dataset.
# 2.6: Output: Function loaded.
from sklearn.datasets import make_moons

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports neural network classifier.
# 2.2: Why it is used: To build the shallow neural networks.
# 2.6: Output: Class loaded.
from sklearn.neural_network import MLPClassifier

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports Scaler and Splitter.
# 2.2: Why it is used: For data preparation.
# 2.6: Output: Classes/Functions loaded.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports metrics.
# 2.2: Why it is used: To evaluate performance.
# 2.6: Output: Functions loaded.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================================================================
# 1. DATA GENERATION
# ==================================================================================================
print("\n--- 1. Generating Data ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates the 'moons' dataset (two interleaving half circles).
# 2.2: Why it is used: Standard benchmark for non-linear classification.
# 2.3: When to used: Testing non-linear classifiers.
# 2.4: Where to use: Data setup.
# 2.5: How to use: `make_moons(...)`.
# 2.6: Output: X (800, 2), y (800,).

# 3. Arguments Explanation:
#    A. n_samples
#       3.1 What: Number of points.
#       3.2 Why: 800 is sufficient for low noise.
#       3.3 When to use: Always.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: Integer.
#       3.6 Argument Example: 800
#    B. noise
#       3.1 What: Standard deviation of Gaussian noise added to data.
#       3.2 Why: Makes the problem harder/more realistic.
#       3.3 When to use: Difficulty control.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: Float.
#       3.6 Argument Example: 0.25
#    C. random_state
#       3.1 What: Seed for reproducibility.
#       3.2 Why: Consistent results.
#       3.6 Argument Example: 21
X, y = make_moons(n_samples=800, noise=0.25, random_state=21)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Splits data into Train and Test.
# 2.2: Why it is used: To evaluate generalization.
# 2.6: Output: 4 Arrays.

# 3. Arguments Explanation:
#    A. arrays
#       3.1 What: Data to split.
#       3.6 Argument Example: X, y
#    B. test_size
#       3.1 What: Fraction for testing.
#       3.2 Why: 30% requested ("70/30 split").
#       3.6 Argument Example: 0.3
#    C. random_state
#       3.1 What: Seed.
#       3.6 Argument Example: 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==================================================================================================
# 2. PREPROCESSING
# ==================================================================================================
print("\n--- 2. Preprocessing ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates a scaler object.
# 2.2: Why it is used: Neural Networks converge much better with zero-mean, unit-variance data.
# 2.6: Output: Scaler object.
scaler = StandardScaler()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Fits scaler to train data and transforms it.
# 2.2: Why it is used: To learn mean/std from Training set only.
# 2.6: Output: Scaled training array.
X_train_scaled = scaler.fit_transform(X_train)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Transforms test data using train statistics.
# 2.2: Why it is used: To apply consistent scaling without leakage.
# 2.6: Output: Scaled test array.
X_test_scaled = scaler.transform(X_test)

# ==================================================================================================
# 3. MODEL TRAINING
# ==================================================================================================
print("\n--- 3. Training Models ---")

# Model 1: Sigmoid
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes MLP with Logistic (Sigmoid) activation.
# 2.6: Output: MLPClassifier Object.

# 3. Arguments Explanation:
#    A. hidden_layer_sizes
#       3.1 What: Architecture (2 layers of 20 nodes).
#       3.2 Why: Shallow network sufficient for moons.
#       3.6 Argument Example: (20, 20)
#    B. activation
#       3.1 What: Function to use at hidden nodes.
#       3.2 Why: 'logistic' is the classic sigmoid function ($1 / (1+e^{-x})$).
#       3.6 Argument Example: 'logistic'
#    C. max_iter
#       3.1 What: Maximum epochs.
#       3.2 Why: Limit training time.
#       3.6 Argument Example: 300
#    D. random_state
#       3.1 What: Seed.
#       3.6 Argument Example: 42
mlp_sigmoid = MLPClassifier(hidden_layer_sizes=(20, 20), activation='logistic', max_iter=300, random_state=42)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Trains the Sigmoid model.
# 2.6: Output: Fitted model.
mlp_sigmoid.fit(X_train_scaled, y_train)

# Model 2: ReLU
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes MLP with ReLU activation.
# 2.6: Output: MLPClassifier Object.

# 3. Arguments Explanation:
#    A. activation
#       3.1 What: Rectified Linear Unit ($max(0, x)$).
#       3.2 Why: Solves vanishing gradient problem; usually converges faster.
#       3.6 Argument Example: 'relu'
mlp_relu = MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', max_iter=300, random_state=42)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Trains the ReLU model.
# 2.6: Output: Fitted model.
mlp_relu.fit(X_train_scaled, y_train)

# ==================================================================================================
# 4. EVALUATION
# ==================================================================================================
print("\n--- 4. Evaluation ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Predicts with Sigmoid model.
# 2.6: Output: Array of labels.
preds_sigmoid = mlp_sigmoid.predict(X_test_scaled)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Predicts with ReLU model.
# 2.6: Output: Array of labels.
preds_relu = mlp_relu.predict(X_test_scaled)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Calculates accuracy scores.
# 2.6: Output: Float scores.
acc_sigmoid = accuracy_score(y_test, preds_sigmoid)
acc_relu = accuracy_score(y_test, preds_relu)

print(f"Sigmoid Accuracy: {acc_sigmoid:.4f}")
print(f"ReLU Accuracy:    {acc_relu:.4f}")

print("\nSigmoid Confusion Matrix:")
# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints confusion matrix.
# 2.6: Output: 2x2 Matrix.
print(confusion_matrix(y_test, preds_sigmoid))

print("\nReLU Confusion Matrix:")
# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints confusion matrix.
# 2.6: Output: 2x2 Matrix.
print(confusion_matrix(y_test, preds_relu))

# ==================================================================================================
# 5. VISUALIZATION
# ==================================================================================================
print("\n--- 5. Plotting Loss Curves ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates plot figure.
# 2.6: Output: Figure object.
plt.figure(figsize=(10, 6))

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots loss curve for Sigmoid model.
# 2.2: Why it is used: `loss_curve_` attribute stores loss value per epoch during training.
# 2.6: Output: Blue line on plot.
plt.plot(mlp_sigmoid.loss_curve_, label='Sigmoid (Logistic)', color='blue')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots loss curve for ReLU model.
# 2.6: Output: Red line on plot.
plt.plot(mlp_relu.loss_curve_, label='ReLU', color='red')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds title and labels.
plt.title('Convergence Comparison: Sigmoid vs ReLU')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds legend and grid.
plt.legend()
plt.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Saves plot.
plt.savefig('loss_comparison.png')
print("Saved plot to loss_comparison.png")
