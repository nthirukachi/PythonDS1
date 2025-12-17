"""
# Problem Statement:
# Implement a Perceptron from Scratch [CODING]
# Build a perceptron classifier from scratch using only NumPy to classify whether students will pass or fail based on study hours and attendance percentage.
# This implementation uses a functional approach (no classes).
#
# Steps to solve the problem:
# 1. Define Helper Functions:
#    - `step_function(z)`: Activation function (returns 1 if input >= 0, else 0).
#    - `train_perceptron(X, y, learning_rate, epochs)`:
#       - Initialize weights and bias to zeros.
#       - Loop through epochs.
#       - For each sample, calculate prediction, error, and update weights/bias.
#       - Return learned weights, bias, and error history.
#    - `predict_perceptron(X, weights, bias)`: Output class labels for new data.
# 2. Generate Synthetic Data:
#    - Create valid relationships between Study Hours, Attendance, and Pass/Fail labels.
# 3. Train the Model:
#    - Call `train_perceptron` with training data.
# 4. Visualization:
#    - Plot training data points colored by class.
#    - Calculate and plot the decision boundary line.
#    - Plot the number of misclassifications (errors) per epoch to verify convergence.
# 5. Testing and Prediction:
#    - Predict outcomes for specific test cases (Student A, B, C).
#    - Output the predictions.
#
# Expected Output:
# - Trained weights and bias.
# - Two plots: Decision Boundary and Error Convergence.
# - Printed predictions for the test students (e.g., Student A: Pass).
# - Printed accuracy on the training set.
"""

import numpy as np # Import numpy for numerical operations (matrix/vector math)
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs

# What: Seed the random number generator
# Why: To ensure reproducibility of the random numbers generated.
# When: At the start of the script before generating random data.
# How: np.random.seed(42) - 42 is an arbitrary fixed number.
np.random.seed(42)

# What: Step activation function
# Why: To convert the continuous linear combination (z) into a binary class label (0 or 1).
# When: Used during prediction and training to decide the output.
# Arguments:
#   z (float or array): The input value(s) (dot product of weights and input + bias).
# Returns: 1 if z >= 0, else 0
def step_function(z):
    return np.where(z >= 0, 1, 0) # Apply threshold at 0

# What: Train the Perceptron algorithm
# Why: To learn the optimal weights and bias that separate the classes.
# When: Called with training data features (X) and labels (y).
# Arguments:
#   X (array-like): Feature matrix of shape (n_samples, n_features).
#   y (array-like): Target vector of shape (n_samples,).
#   learning_rate (float): The step size for weight updates (0.0 to 1.0). Default 0.01.
#   epochs (int): The number of passes over the training dataset. Default 100.
# Returns:
#   weights (array): Optimized weights.
#   bias (float): Optimized bias.
#   errors_history (list): Number of misclassifications per epoch.
def train_perceptron(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape # Get dimensions of the data
    
    # What: Initialize parameters
    # Why: Start learning from a neutral point (zeros).
    # When: At the beginning of training.
    weights = np.zeros(n_features) # Initialize weights to zeros
    bias = 0.0                     # Initialize bias to 0.0
    errors_history = []            # List to store error counts per epoch

    # Loop over the dataset 'epochs' times
    for _ in range(epochs):
        errors = 0 # Track misclassifications in this epoch
        
        # Loop over each individual sample
        for idx, x_i in enumerate(X):
            # Calculate linear output: z = w * x + b
            linear_output = np.dot(x_i, weights) + bias
            
            # Apply activation function to get prediction
            y_predicted = step_function(linear_output)
            
            # Calculate update term: (target - prediction)
            # If target == prediction, update is 0.
            # If target=1, pred=0, update is 1 (increase weights).
            # If target=0, pred=1, update is -1 (decrease weights).
            update = learning_rate * (y[idx] - y_predicted)
            
            # Update weights and bias
            # New_weight = Old_weight + (learning_rate * error * input_value)
            weights += update * x_i
            bias += update
            
            # If there was an update (update != 0), count as an error
            if update != 0:
                errors += 1
        
        errors_history.append(errors) # Store error count for this epoch
    
    return weights, bias, errors_history

# What: Make predictions on new data
# Why: To classify unseen or test data points.
# When: After training the model.
# Arguments:
#   X (array-like): Data to predict on.
#   weights (array): Trained weights.
#   bias (float): Trained bias.
def predict_perceptron(X, weights, bias):
    # linear_output = w1*x1 + w2*x2 + ... + b
    linear_output = np.dot(X, weights) + bias
    # Return class label (0 or 1)
    return step_function(linear_output)

# --- Main Execution ---

# 1. Generate Synthetic Data
# What: Create 100 samples with generic randomness
# Why: To simulate a real-world student dataset.
# When: First step of the workflow.
n_samples = 100
# Study hours: Random integer between 0 and 100
study_hours = np.random.randint(0, 100, n_samples)
# Attendance: Random integer between 40 and 100
attendance = np.random.randint(40, 100, n_samples)

# Create labels based on a logical rule: Pass if study_hours + 0.5*attendance > 75
# calculated_score is the linear combination that determines the ground truth
calculated_score = study_hours + 0.5 * attendance
labels = (calculated_score > 75).astype(int) # 1 for Pass, 0 for Fail

# Stack features into a single matrix (100 rows, 2 columns)
# Note: Scaling features (0-100 -> 0-1) to help the Perceptron converge faster and better.
X_train = np.column_stack([study_hours, attendance]) / 100.0
y_train = labels

print(f"Data generated. Feature shape: {X_train.shape}, Label shape: {y_train.shape}")

# 2. Train the Perceptron
# What: Call the training function
# Why: To fit the model to our generated data.
# When: After data preparation.
# Using Learning Rate=0.1 and Epochs=1000 for better convergence on this data
final_weights, final_bias, error_history = train_perceptron(X_train, y_train, learning_rate=0.1, epochs=1000)

print(f"Training complete. Weights: {final_weights}, Bias: {final_bias}")

# 3. Visualization

# --- Plot 1: Decision Boundary ---
plt.figure(figsize=(10, 6)) # Create a new figure with specific size

# Scatter plot for class 0 (Fail) - Red points
# X_train[y_train == 0][:, 0] selects 'study_hours' where label is 0
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Fail (0)')
# Scatter plot for class 1 (Pass) - Blue points
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Pass (1)')

# Plotting the Decision Boundary Line
# The boundary is where w1*x1 + w2*x2 + b = 0
# We can express x2 (attendance) as a function of x1 (study_hours):
# w2*x2 = -w1*x1 - b
# x2 = -(w1*x1 + b) / w2
x1_min, x1_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
x1_values = np.linspace(x1_min, x1_max, 100) # Generate 100 points along x-axis
# Calculate corresponding y values (x2)
x2_values = -(final_weights[0] * x1_values + final_bias) / final_weights[1]

# Plot the line
plt.plot(x1_values, x2_values, color='green', linestyle='--', label='Decision Boundary')

plt.title('Perceptron Decision Boundary: Study Hours vs Attendance')
plt.xlabel('Study Hours (Scaled)')
plt.ylabel('Attendance % (Scaled)')
plt.legend()
plt.grid(True)
plt.savefig('c:/nagpython/demouv/Ex/Perceptron/decision_boundary_plot.png') # Save plot to file
print("Saved decision_boundary_plot.png")

# --- Plot 2: Convergence ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_history) + 1), error_history, marker='o')
plt.title('Perceptron Convergence: Errors vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.grid(True)
plt.savefig('c:/nagpython/demouv/Ex/Perceptron/convergence_plot.png') # Save plot to file
print("Saved convergence_plot.png")

# 4. Test Predictions
# Test cases provided in requirement
# Student A: 80 study hours, 90% attendance
# Student B: 30 study hours, 60% attendance
# Student C: 50 study hours, 85% attendance
test_data = np.array([
    [80, 90],
    [30, 60],
    [50, 85]
])

# Normalize the test data as well using the same scale
test_data_scaled = test_data / 100.0

print("\n--- Test Predictions ---")
predictions = predict_perceptron(test_data_scaled, final_weights, final_bias)
student_names = ['Student A', 'Student B', 'Student C']

for name, features, pred in zip(student_names, test_data, predictions):
    status = "Pass" if pred == 1 else "Fail"
    print(f"{name} (Study: {features[0]}, Attendance: {features[1]}%): Prediction -> {status} ({pred})")

# Check accuracy on training set
train_predictions = predict_perceptron(X_train, final_weights, final_bias)
accuracy = np.mean(train_predictions == y_train) * 100
print(f"\nModel Accuracy on Training Data: {accuracy:.2f}%")
