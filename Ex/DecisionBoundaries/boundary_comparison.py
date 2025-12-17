"""
# Problem Statement:
# Visualize Decision Boundaries with Different Activations [CODING]
# Build a simple neural network and visualize how different activation functions create different decision boundaries on a non-linearly separable dataset.
# Dataset: Use sklearn's make_moons dataset.
#
# Steps:
# 1. Load Data:
#    - Generate 'make_moons' dataset with noise to create a non-linear classification problem.
# 2. Define Model Configurations:
#    - Set up 3 MLPClassifiers with the same architecture (1 hidden layer, 8 neurons) but different activations: ReLU, Logistic (Sigmoid), Tanh.
# 3. Model Training:
#    - Iterate through each configuration, training the model on the full dataset.
# 4. Visualization:
#    - Create a meshgrid covering the feature space.
#    - Predict class labels for every point in the grid to define decision regions.
#    - Plot decision boundaries and overlay the training data points.
# 5. Analysis:
#    - Calculate training accuracy for each model.
#    - Compare how the choice of activation function changes the shape of the decision boundary.
#
# Expected Output:
# - A single figure file with 3 subplots showing the decision boundaries.
# - Console output containing the accuracy of each model.
"""

import numpy as np # Import numpy for matrix operations
import matplotlib.pyplot as plt # Import matplotlib for plotting
from sklearn.datasets import make_moons # Import dataset generator
from sklearn.neural_network import MLPClassifier # Import Neural Network Classifier from sklearn
from matplotlib.colors import ListedColormap # Helper for coloring plots

# --- 1. Data Generation ---
# What: Generate synthetic non-linear data
# Why: To test if the neural network can learn non-linear patterns (moons shape).
# When: At the start of the experiment.
# Arguments:
#   n_samples=300: Total number of data points.
#   noise=0.2: Standard deviation of Gaussian noise added to the data (makes it harder/more realistic).
#   random_state=42: Seed for reproducibility.
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# --- 2. Visualization Setup ---
# Define the plot boundaries based on data range
# What: Find min/max of features to define plot limits
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# What: Create a meshgrid
# Why: To predict values for every single pixel/point in the background of the plot, creating a solid color region.
# How: np.meshgrid creates coordinate matrices from coordinate vectors.
# step=0.02: Resolution of the grid.
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# List of activation functions to compare
activations = ['relu', 'logistic', 'tanh']
names = ["ReLU", "Sigmoid (Logistic)", "Tanh"]

# Create a figure with 3 subplots in a row
figure = plt.figure(figsize=(18, 5))

print(f"{'Activation':<15} | {'Training Accuracy':<10}")
print("-" * 35)

# --- 3. Training and Plotting Loop ---

# Enumerate through activations: idx is 0,1,2; name is 'relu', etc.
for i, (activation, name) in enumerate(zip(activations, names)):
    
    # What: Initialize the MLP (Multi-Layer Perceptron) Classifier
    # Why: To create a neural network model.
    # When: Inside the loop for each activation type.
    # Arguments:
    #   hidden_layer_sizes=(8,): Architecture of 1 hidden layer with 8 neurons.
    #   activation=activation: The specific function we are testing (relu, logistic, tanh).
    #   max_iter=1000: Maximum number of epochs (iterations) to run training.
    #   random_state=42: Ensures weight initialization is consistent across models for fair comparison.
    clf = MLPClassifier(hidden_layer_sizes=(8,), activation=activation, max_iter=2000, random_state=42)
    
    # What: Train the model
    # Why: To learn the weights and biases from the data.
    clf.fit(X, y)
    
    # What: Calculate Accuracy
    # Why: To quantitatively evaluate performance.
    score = clf.score(X, y)
    print(f"{name:<15} | {score:.2%}")
    
    # --- Plotting ---
    
    # Create subplot axis (1 row, 3 columns, current index i+1)
    ax = plt.subplot(1, 3, i + 1)
    
    # What: Predict for the entire meshgrid
    # Why: Top determine the class (0 or 1) for every point in the plot background.
    # ravel(): Flattens the 2D grid arrays into 1D arrays to be stacked.
    # predict_proba: Returns probability [prob_class0, prob_class1]. we take column 1 (prob_class1).
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # For some other classifiers
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] # Probability of positive class

    # Reshape prediction results back to meshgrid shape for contour plotting
    Z = Z.reshape(xx.shape)
    
    # What: Draw the decision boundary contour
    # Why: To visualize the region separation.
    # cm: Colormap (Red-Blue)
    # alpha: Transparency
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # What: Plot the training points
    # Why: To see how well the regions fit the actual data.
    # c=y: Color points based on their true label.
    # edgecolors='k': Black outline for visibility.
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k')
    
    # Formatting the plot
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(()) # Remove x-axis ticks for cleaner look
    ax.set_yticks(()) # Remove y-axis ticks
    
    # Add Title with Name and Accuracy
    ax.set_title(f"{name}\nAcc: {score:.2%}")

# Save the complete figure
plt.tight_layout()
plt.savefig('c:/nagpython/demouv/Ex/DecisionBoundaries/decision_boundaries.png')
print("\nSaved plot to c:/nagpython/demouv/Ex/DecisionBoundaries/decision_boundaries.png")
