"""
# Problem Statement:
# Implement Leaky ReLU and Compare with Standard ReLU [CODING]
# Investigate the dying ReLU problem by implementing both ReLU and Leaky ReLU, then comparing their behavior.
#
# Steps to solve the problem:
# 1. Data Generation:
#    - Generate random synthetic data with 10 features.
#    - Create binary labels based on a linear combination of features.
# 2. Define Activation Functions:
#    - Implement `relu` and `leaky_relu` (forward pass).
#    - Implement their derivatives `relu_derivative` and `leaky_relu_derivative` (backward pass).
# 3. Implement Neural Network Class:
#    - Initialize weights and biases.
#    - Implement `forward` propagation to compute activations.
#    - Implement `backward` propagation to compute gradients and update weights using Gradient Descent.
# 4. Training:
#    - Train Model 1 with ReLU for 200 epochs.
#    - Train Model 2 with Leaky ReLU for 200 epochs.
#    - Record training loss history.
# 5. Analysis:
#    - Visualize the loss curves to compare convergence.
#    - Calculate the percentage of "dead neurons" (neurons that never output value > 0 during a pass).
#    - Compare final accuracy.
#
# Expected Output:
# - A plot comparing Training Loss over epochs.
# - Console output with Accuracy and Dead Neuron percentage for both models.
"""

import numpy as np # Import numpy for matrix operations
import matplotlib.pyplot as plt # Import matplotlib for plotting

# What: Set seed
# Why: Reproducibility.
np.random.seed(42)

# --- 1. Data Generation ---
# What: Generate 1000 samples, 10 features
# Why: Enough data to train a small network.
X_train = np.random.randn(1000, 10)
# Create labels: A somewhat complex linear boundary
y_train = (X_train[:, 0] + X_train[:, 1] - X_train[:, 2] > 0).astype(int).reshape(-1, 1)

# --- 2. Activation Functions ---

# What: ReLU (Rectified Linear Unit)
# Why: Standard activation. Returns x if x>0, else 0.
def relu(z):
    return np.maximum(0, z)

# What: Derivative of ReLU
# Why: For backprop. 1 if z>0, else 0.
def relu_derivative(z):
    return (z > 0).astype(float)

# What: Leaky ReLU
# Why: Allows a small gradient when z<0 to prevent "dying" neurons.
# Arguments: alpha (slope for negative values, default 0.01)
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# What: Derivative of Leaky ReLU
# Why: For backprop. 1 if z>0, else alpha.
def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# What: Sigmoid (for output layer)
# Why: To squash output between 0 and 1 for binary classification probability.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# What: Sigmoid Derivative
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# --- 3. Neural Network Class ---

class NeuralNetwork:
    """
    A simple 2-layer Neural Network (Input -> Hidden -> Output).
    Supports 'relu' or 'leaky_relu' for the hidden layer.
    """
    
    # What: Initialize the network
    # Arguments:
    #   input_size: Number of features (10).
    #   hidden_size: Number of hidden neurons (20).
    #   output_size: Number of output neurons (1).
    #   activation_type: 'relu' or 'leaky_relu'.
    def __init__(self, input_size, hidden_size, output_size, activation_type='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_type = activation_type
        
        # Initialize weights with He Initialization (good for ReLU)
        # Why: Breaks symmetry and keeps variance controlled.
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []

    # What: Forward Propagation
    # Why: Compute the output prediction given an input X.
    # Returns: Final output (A2)
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        
        # Apply hidden activation
        if self.activation_type == 'relu':
            self.A1 = relu(self.Z1)
        elif self.activation_type == 'leaky_relu':
            self.A1 = leaky_relu(self.Z1)
            
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2) # Output layer is always Sigmoid for binary class
        return self.A2

    # What: Backward Propagation
    # Why: Calculate gradients of the loss function with respect to weights.
    # Arguments: 
    #   X: Input features.
    #   y: True labels.
    #   learning_rate: Step size for update.
    def backward(self, X, y, learning_rate):
        m = X.shape[0] # Number of samples
        
        # Calculate Output Layer error (dZ2)
        # Loss = Binary Cross Entropy. Derivative wrt Z2 simplifies to (Prediction - Truth)
        dZ2 = self.A2 - y
        
        # Gradients for W2, b2
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Calculate Hidden Layer error (dA1 -> dZ1)
        dA1 = np.dot(dZ2, self.W2.T)
        
        # Derivative of hidden activation
        if self.activation_type == 'relu':
            dZ1 = dA1 * relu_derivative(self.Z1)
        elif self.activation_type == 'leaky_relu':
            dZ1 = dA1 * leaky_relu_derivative(self.Z1)
            
        # Gradients for W1, b1
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update Weights (Gradient Descent)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # What: Train the model
    # Why: Loop forward and backward passes to optimize weights.
    def train(self, X, y, epochs=200, learning_rate=0.01):
        for i in range(epochs):
            # Forward pass
            output = self.forward(X)
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Record Loss (Binary Cross Entropy)
            # epsilon added to log to prevent log(0)
            epsilon = 1e-15
            loss = -np.mean(y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon))
            self.loss_history.append(loss)

# --- 4. Count Dead Neurons ---
# What: Helper function to count dead neurons
# Why: To investigate if neurons are stuck at 0 output.
def count_dead_neurons(model, X):
    # Perform a forward pass to get hidden activations (A1)
    # Z1 is the pre-activation input. Dead neuron = Z1 is always negative (for ReLU) -> A1 is always 0.
    # But for counting "dead" in terms of output, we check if activation is 0 for ALL samples.
    model.forward(X)
    hidden_activations = model.A1 # Shape (samples, hidden_size)
    
    # Check if a neuron (column) has 0 activation for ALL input samples
    # For ReLU: This is a "dead neuron".
    # For Leaky ReLU: Values shouldn't be exactly 0 unless input is 0, they will be small negative numbers.
    # So strictly "dead" (0 output) is mostly a ReLU concept.
    # We will count how many neurons have <= 0 sum of activity (should only happen for ReLU if truly dead).
    
    dead_count = 0
    for j in range(model.hidden_size):
        neuron_activity = hidden_activations[:, j]
        # If max activity is 0, it never fired positively.
        if np.max(neuron_activity) == 0:
            dead_count += 1
            
    return dead_count, model.hidden_size

# --- 5. Execution ---

epochs = 200
learning_rate = 0.01

# Train ReLU Model
print("Training ReLU Model...")
nn_relu = NeuralNetwork(input_size=10, hidden_size=20, output_size=1, activation_type='relu')
nn_relu.train(X_train, y_train, epochs, learning_rate)

# Train Leaky ReLU Model
print("Training Leaky ReLU Model...")
nn_leaky = NeuralNetwork(input_size=10, hidden_size=20, output_size=1, activation_type='leaky_relu')
nn_leaky.train(X_train, y_train, epochs, learning_rate)

# --- Analysis ---

# Accuracy
pred_relu = (nn_relu.forward(X_train) > 0.5).astype(int)
acc_relu = np.mean(pred_relu == y_train) * 100

pred_leaky = (nn_leaky.forward(X_train) > 0.5).astype(int)
acc_leaky = np.mean(pred_leaky == y_train) * 100

print(f"\nResults:")
print(f"{'Model':<15} | {'Accuracy':<10} | {'Dead Neurons':<15}")
print("-" * 50)

dead_relu, total = count_dead_neurons(nn_relu, X_train)
dead_leaky, _ = count_dead_neurons(nn_leaky, X_train)

print(f"{'ReLU':<15} | {acc_relu:.2f}%     | {dead_relu}/{total} ({dead_relu/total:.1%})")
print(f"{'Leaky ReLU':<15} | {acc_leaky:.2f}%     | {dead_leaky}/{total} ({dead_leaky/total:.1%})")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(nn_relu.loss_history, label=f'Standard ReLU (Acc: {acc_relu:.1f}%)', color='red')
plt.plot(nn_leaky.loss_history, label=f'Leaky ReLU (Acc: {acc_leaky:.1f}%)', color='blue', linestyle='--')
plt.title('Training Loss Comparison: ReLU vs Leaky ReLU')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig('c:/nagpython/demouv/Ex/LeakyReluComparison/loss_comparison.png')
print("\nSaved loss comparison plot to c:/nagpython/demouv/Ex/LeakyReluComparison/loss_comparison.png")
