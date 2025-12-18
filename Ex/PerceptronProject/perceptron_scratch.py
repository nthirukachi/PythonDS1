"""
====================================================================================================
1. PROBLEM STATEMENT:
Build a perceptron from scratch on a clearly separable dataset and analyse its learning dynamics.

We need to:
1.  Generate a synthetic dataset that is linearly separable (using make_classification).
2.  Implement the Perceptron learning algorithm using only NumPy (no classes, just functions).
3.  Train the model for 40 epochs, shuffling data each time to ensure robust learning.
4.  Track the accuracy at each epoch to visualize the learning curve.
5.  Count the total number of weight updates (mistakes corrected).
6.  Visualize the final decision boundary separating the two classes.

STEPS TO SOLVE THE PROBLEM:
1.  Data Setup: Generate 600 samples, 2 features, cleanly separated. Split into Train/Test (80/20).
2.  Initialization: Create a weight vector and bias initialized to zero within the training function.
3.  Training Function:
    -   Input: Data (X, y), Learning Rate, Epochs.
    -   Loop: Run 40 epochs.
    -   Shuffle: Randomize training data order.
    -   Update: If error != 0, adjust weights: w = w + lr * error * x.
    -   Return: Final weights, bias, history, and update count.
4.  Prediction Function:
    -   Input: Data (X), trained weights, trained bias.
    -   Output: 0 or 1 based on dot product.
5.  Evaluation: Compute accuracy on training set after each epoch.
6.  Testing: Evaluate final accuracy on held-out test set.
7.  Plotting: Draw the separating line and the accuracy history.

EXPECTED OUTPUT:
-   Console logs showing accuracy improving over epochs.
-   Final Test Accuracy >= 0.95.
-   A plot showing two distinct clusters separated by a straight line.
-   Total update count (indicating how many mistakes were fixed).
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the NumPy library.
# 2.2: Why it is used: Essential for efficient matrix and vector operations (like dot products).
# 2.3: When to used: In almost every data science or machine learning script.
# 2.4: Where to use: At the very top of the script.
# 2.5: How to use: standard import `import numpy as np`.
# 2.6: Output: None (Library loaded into namespace).
import numpy as np

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports Pyplot from Matplotlib.
# 2.2: Why it is used: To create visualizations (scatter plots, line charts) of data and results.
# 2.3: When to used: When visual analysis is required.
# 2.4: Where to use: Global scope.
# 2.5: How to use: standard import `import matplotlib.pyplot as plt`.
# 2.6: Output: None (Library loaded into namespace).
import matplotlib.pyplot as plt

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the `make_classification` function from sklearn.datasets.
# 2.2: Why it is used: To generate a synthetic dataset with controlled properties (separable classes).
# 2.3: When to used: For testing algorithms when real data is not available.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.datasets import ...`.
# 2.6: Output: None (Function imported).
from sklearn.datasets import make_classification

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the `train_test_split` function.
# 2.2: Why it is used: To split the data into training and testing sets, ensuring unbiased evaluation.
# 2.3: When to used: In every supervised learning project.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.model_selection import ...`.
# 2.6: Output: None (Function imported).
from sklearn.model_selection import train_test_split

# ==================================================================================================
# 1. DATA GENERATION
# ==================================================================================================
print("\n--- 1. Generating Data ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates a synthetic binary classification dataset.
# 2.2: Why it is used: To enable us to test the Perceptron on a "clearly separable" problem.
# 2.3: When to used: Initial data setup.
# 2.4: Where to use: Variable assignment.
# 2.5: How to use: Call `make_classification` with specific arguments.
# 2.6: Output: Two NumPy arrays: X (samples x features) and y (samples).

# 3. Arguments Explanation:
#    A. n_samples
#       3.1 What: Number of data points.
#       3.2 Why: Problem requested 600 samples.
#       3.3 When to use: Always.
#       3.4 Where to use: Argument 1.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 600
#    B. n_features
#       3.1 What: Number of features per sample.
#       3.2 Why: 2 features are easy to plot on a 2D plane.
#       3.3 When to use: To define dimensionality.
#       3.4 Where to use: Argument 2.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 2
#    C. n_informative
#       3.1 What: Number of features that actually contribute to the class.
#       3.2 Why: Both features should matter.
#       3.3 When to use: To control signal.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Integer <= n_features.
#       3.6 Sample Example: 2
#    D. n_redundant
#       3.1 What: Number of features that are linear combinations of informative ones.
#       3.2 Why: We want a clean, simple dataset.
#       3.3 When to use: To add noise/complexity.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 0
#    E. class_sep
#       3.1 What: The factor multiplying the hypercube size. Larger = more separated.
#       3.2 Why: 1.6 ensures the classes do not overlap, satisfying "clearly separable".
#       3.3 When to use: To adjust difficulty.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Float.
#       3.6 Sample Example: 1.6
#    F. random_state
#       3.1 What: Seed for the random number generator.
#       3.2 Why: Ensures we get the exact same dataset every time we run the code.
#       3.3 When to use: Reproducible research.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 7
X, y = make_classification(
    n_samples=600,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    class_sep=1.6,
    random_state=7,
)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Splits the dataset into Training (80%) and Testing (20%) sets.
# 2.2: Why it is used: To meet the requirement "test accuracy >= 0.95 on a 20 percent holdout".
# 2.3: When to used: After data generation.
# 2.4: Where to use: Variable assignment.
# 2.5: How to use: `train_test_split(X, y, ...)`.
# 2.6: Output: Four arrays: X_train, X_test, y_train, y_test.

# 3. Arguments Explanation:
#    A. arrays (X, y)
#       3.1 What: The feature and label arrays to split.
#       3.2 Why: Input data.
#       3.3 When to use: Always.
#       3.4 Where to use: Positional args.
#       3.5 How to use: Pass variables.
#       3.6 Sample Example: X, y
#    B. test_size
#       3.1 What: Proportion of the dataset to include in the test split.
#       3.2 Why: 0.2 means 20% for testing, 80% for training.
#       3.3 When to use: Always.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Float between 0.0 and 1.0.
#       3.6 Sample Example: 0.2
#    C. random_state
#       3.1 What: Seed for the random shuffling before split.
#       3.2 Why: Ensures the split is the same every time.
#       3.3 When to use: Reproducibility.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ==================================================================================================
# 2. FUNCTIONAL IMPLEMENTATION
# ==================================================================================================
print("\n--- 2. Helper Functions ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines a prediction function.
# 2.2: Why it is used: To encapsulate the Perceptron prediction logic ($z = w \cdot x + b > 0$).
# 2.3: When to used: During training (to check for errors) and evaluation.
# 2.4: Where to use: Function definition.
# 2.5: How to use: `pred = predict(X, w, b)`.
# 2.6: Output: A NumPy array of predictions (0s and 1s).

# 3. Arguments Explanation:
#    A. X
#       3.1 What: Input features (samples x features).
#       3.2 Why: The data we want to classify.
#       3.3 When to use: Training and Testing.
#       3.4 Where to use: Argument 1.
#       3.5 How to use: NumPy Array.
#       3.6 Sample Example: X_train
#    B. weights
#       3.1 What: The weight vector ($w_1, w_2$).
#       3.2 Why: Defines the orientation of the decision boundary.
#       3.3 When to use: Always.
#       3.4 Where to use: Argument 2.
#       3.5 How to use: NumPy Array.
#       3.6 Sample Example: np.array([0.5, -0.5])
#    C. bias
#       3.1 What: The bias term ($b$).
#       3.2 Why: Shifts the decision boundary from the origin.
#       3.3 When to use: Always.
#       3.4 Where to use: Argument 3.
#       3.5 How to use: Float.
#       3.6 Sample Example: 0.1
def predict(X, weights, bias):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Computes the dot product of X and weights, adding bias.
    # 2.2: Why it is used: This gives the "linear score" or "activation potential".
    # 2.3: When to used: First step of prediction.
    # 2.4: Where to use: Inside function.
    # 2.5: How to use: `np.dot` operator or function.
    # 2.6: Output: Array of float values.
    linear_output = np.dot(X, weights) + bias
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Applies the Heaviside Step Function.
    # 2.2: Why it is used: To convert continuous scores into binary class labels (0 or 1).
    # 2.3: When to used: Final step of prediction.
    # 2.4: Where to use: Return statement.
    # 2.5: How to use: `np.where(condition, value_if_true, value_if_false)`.
    # 2.6: Output: Array of 0s and 1s.
    return np.where(linear_output > 0, 1, 0)


# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines the main training function.
# 2.2: Why it is used: To iteratively update weights based on errors (The Perceptron Algorithm).
# 2.3: When to used: Model training phase.
# 2.4: Where to use: Function definition.
# 2.5: How to use: `w, b, hist, count = train_perceptron(X, y)`.
# 2.6: Output: Tuple containing (weights, bias, history list, update count).

# 3. Arguments Explanation:
#    A. X
#       3.1 What: Training features.
#       3.2 Why: Data to learn from.
#       3.3 When to use: Always.
#       3.4 Where to use: Argument 1.
#       3.5 How to use: NumPy Array.
#       3.6 Sample Example: X_train
#    B. y
#       3.1 What: Training labels.
#       3.2 Why: Ground truth to correct mistakes.
#       3.3 When to use: Always.
#       3.4 Where to use: Argument 2.
#       3.5 How to use: NumPy Array.
#       3.6 Sample Example: y_train
#    C. learning_rate
#       3.1 What: The size of the weight update step ($\eta$).
#       3.2 Why: Controls how fast the model adapts. Too big = unstable; too small = slow.
#       3.3 When to use: Hyperparameter tuning.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Float.
#       3.6 Sample Example: 0.01
#    D. epochs
#       3.1 What: The number of full passes through the dataset.
#       3.2 Why: Ensures the model has enough opportunities to converge.
#       3.3 When to use: Hyperparameter tuning.
#       3.4 Where to use: Keyword Argument.
#       3.5 How to use: Integer.
#       3.6 Sample Example: 40
def train_perceptron(X, y, learning_rate=0.01, epochs=40):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Unpacks the shape of X.
    # 2.2: Why it is used: We need `n_samples` for looping and `n_features` for initialization.
    # 2.6: Output: Tuple assignment (600, 2).
    n_samples, n_features = X.shape
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Initializes weights to a vector of zeros.
    # 2.2: Why it is used: Standard starting point.
    # 2.6: Output: Array `[0. 0.]`.
    weights = np.zeros(n_features)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Initializes bias to zero.
    # 2.2: Why it is used: Standard starting point.
    # 2.6: Output: Integer 0.
    bias = 0
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Initializes empty list for tracking accuracy.
    # 2.6: Output: Empty list `[]`.
    history = []
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Initializes counter for updates.
    # 2.6: Output: Integer 0.
    total_updates = 0
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Loops for the specified number of epochs.
    # 2.2: Why it is used: To repeat the learning process ("Train for at least 40 epochs").
    # 2.6: Output: Iterator.
    for epoch in range(epochs):
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Generates a random permutation of indices from 0 to n_samples-1.
        # 2.2: Why it is used: To shuffle the data order (Requirement: "shuffling each epoch").
        # 2.6: Output: Array of shuffled integers.
        indices = np.random.permutation(n_samples)
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Reorders X according to the shuffled indices.
        # 2.6: Output: Shuffled feature array.
        X_shuffled = X[indices]
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Reorders y according to the shuffled indices.
        # 2.6: Output: Shuffled label array.
        y_shuffled = y[indices]
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Iterates through each sample one by one.
        # 2.2: Why it is used: Perceptron is an "online" or "stochastic" algorithm; it updates on single samples.
        # 2.6: Output: Iterator.
        for i in range(n_samples):
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Extracts the current sample features.
            # 2.6: Output: Single data point array.
            x_i = X_shuffled[i]
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Extracts the current sample label.
            # 2.6: Output: Integer (0 or 1).
            target = y_shuffled[i]
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Calculates linear activation.
            # 2.6: Output: Float scalar.
            linear_output = np.dot(x_i, weights) + bias
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Applies threshold (1 if >0 else 0).
            # 2.6: Output: Integer prediction.
            y_predicted = 1 if linear_output > 0 else 0
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Calculates the Update term.
            # 2.2: Why it is used: Determines direction and magnitude of correction.
            # 2.6: Output: Float (0.0, 0.01, or -0.01).
            update = learning_rate * (target - y_predicted)
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Checks if an update is needed (error != 0).
            # 2.6: Output: Boolean.
            if update != 0:
                # 2. Detailed Explanation:
                # 2.1: What the line of code does: Increments total update count.
                # 2.6: Output: None (Update variable).
                total_updates += 1
                
                # 2. Detailed Explanation:
                # 2.1: What the line of code does: Updates weights vector.
                # 2.2: Why it is used: To rotate the decision boundary.
                # 2.6: Output: None (Update variable).
                weights += update * x_i
                
                # 2. Detailed Explanation:
                # 2.1: What the line of code does: Updates bias term.
                # 2.2: Why it is used: To shift the decision boundary.
                # 2.6: Output: None (Update variable).
                bias += update
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Predicts on the *entire* original dataset using current weights.
        # 2.2: Why it is used: To calculate accuracy for this epoch.
        # 2.6: Output: Array of predictions.
        preds = predict(X, weights, bias)
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Calculates mean accuracy.
        # 2.2: Why it is used: Metric tracking.
        # 2.6: Output: Float between 0.0 and 1.0.
        acc = np.mean(preds == y)
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Appends accuracy to history list.
        # 2.6: Output: None (List modified).
        history.append(acc)
        
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Returns the final trained parameters and stats.
    # 2.6: Output: Tuple.
    return weights, bias, history, total_updates

# ==================================================================================================
# 3. EXECUTION
# ==================================================================================================
print("\n--- 3. Training Start ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Runs the training function on the training data.
# 2.2: Why it is used: To train our model.
# 2.6: Output: Returns (w, b, acc, updates).
w_final, b_final, acc_history, updates_count = train_perceptron(X_train, y_train, learning_rate=0.01, epochs=40)

print(f"Total Weight Updates: {updates_count}")

# ==================================================================================================
# 4. EVALUATION
# ==================================================================================================
print("\n--- 4. Evaluation ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates predictions for the held-out test set.
# 2.6: Output: 0/1 Array.
test_preds = predict(X_test, w_final, b_final)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Calculates final test accuracy.
# 2.6: Output: Float.
test_accuracy = np.mean(test_preds == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ==================================================================================================
# 5. VISUALIZATION
# ==================================================================================================
print("\n--- 5. Plotting ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates a figure for plotting with size 12x5 inches.
plt.figure(figsize=(12, 5))

# Plot 1: Decision Boundary
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Selects the first subplot in a 1x2 grid.
plt.subplot(1, 2, 1)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates a scatter plot of feature 1 vs feature 2, colored by class.
# 2.6: Output: Plot object.

# 3. Arguments:
#    A. x, y
#       3.1 What: Coordinates.
#       3.6 Example: X[:,0], X[:,1]
#    B. c
#       3.1 What: Color array.
#       3.6 Example: c=y
#    C. cmap
#       3.1 What: Colormap ('bwr' = Blue-White-Red).
#       3.6 Example: 'bwr'
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

# Boundary Line Calculation
# 2. Detailed Explanation:
# 2.1: What the line of code does: Calculates min/max x1 values for plotting line segments.
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates 100 points between min and max x1.
x1_vals = np.linspace(x1_min, x1_max, 100)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Solves for x2 given x1, w, and b ($x_2 = -(w_1 x_1 + b) / w_2$).
# 2.2: Why it is used: To get the y-coordinates of the decision boundary line.
x2_vals = -(w_final[0] * x1_vals + b_final) / w_final[1]

# 2. Detailed Explanation:
# 2.1: What the line of code does: Draws the boundary line in dashed black.
plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='Decision Boundary')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds title/labels/legend.
plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot 2: Learning Curve
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Selects the second subplot.
plt.subplot(1, 2, 2)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots the accuracy history over epochs.
plt.plot(range(1, 41), acc_history, marker='o')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds title/labels/grid.
plt.title('Learning Dynamics (Accuracy per Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adjusts subplots to fit nicely.
plt.tight_layout()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Saves the plot to a PNG file.
plt.savefig('perceptron_analysis.png')

print("Saved plot to perceptron_analysis.png")
