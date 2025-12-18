"""
# =============================================================================
# PROBLEM STATEMENT:
# =============================================================================
# Train an XOR classifier from scratch using manual backpropagation.
# The XOR (Exclusive OR) problem is a classic example that linear classifiers
# cannot solve. It requires a non-linear activation function and a hidden layer.
#
# Inputs: [[0,0], [0,1], [1,0], [1,1]]
# Labels: [0, 1, 1, 0]
#
# STEPS TO SOLVE THE PROBLEM:
# 1.  **Initialization**: Setup the Neural Network architecture (2-input, 2-hidden, 1-output).
#     Initialize weights randomly and biases to zero.
# 2.  **Forward Propagation**: 
#     - Compute Hidden Layer Linear Step (Z1).
#     - Apply Activation (ReLU) -> A1.
#     - Compute Output Layer Linear Step (Z2).
#     - Apply Activation (Sigmoid) -> A2 (Prediction).
# 3.  **Loss Calculation**: Compare Prediction (A2) with True Label (Y) using Binary Cross-Entropy.
# 4.  **Backward Propagation**:
#     - Calculate gradients for Output Layer (dZ2, dW2, db2).
#     - Propagate error to Hidden Layer (dA1).
#     - Calculate gradients for Hidden Layer (dZ1, dW1, db1).
# 5.  **Optimization (Gradient Descent)**: Update weights and biases using the gradients and learning rate.
# 6.  **Gradient Check**: Verify the manual backprop implementation using Finite Difference method.
#
# EXPECTED OUTPUT:
# - Loss should decrease over 5000 iterations (target < 0.02).
# - Predictions should match the XOR truth table with high confidence (> 0.95).
# - Gradient check error should be minimal (< 1e-3).
# =============================================================================
"""

# 2.1 Definition: Import the NumPy library.
# 2.2 Why: NumPy is the fundamental package for scientific computing in Python. It provides support for arrays and matrices.
# 2.3 When: Always used when performing linear algebra, vectorization, or mathematical operations on data.
# 2.4 Where: At the very beginning of the script.
# 2.5 How to use: `import numpy as np` allows accessing functions via `np.function()`.
# 2.6 How it works: Loads the C-optimized math library into memory.
# 2.7 Output: Module object 'np'.
import numpy as np


# 2.1 Definition: Set the random seed.
# 2.2 Why: To ensure reproducibility. Random numbers will be the same every time code is run.
# 2.3 When: Before generating any random numbers.
# 2.4 Where: Global scope or start of execution.
# 2.5 How to use: `np.random.seed(integer_value)`.
# 2.6 How it works: Initializes the pseudo-random number generator state with the given integer.
# 2.7 Output: None (State is updated internally).
np.random.seed(7)


def sigmoid(z):
    """
    3.1 Argument `z`: Scalar or NumPy array.
    3.2 Why: The input value(s) to be squashed.
    3.3 When: During Forward Propagation (Output Layer).
    3.4 Where: Passed from the linear combination of weights and inputs.
    3.5 How to use: `a = sigmoid(z)`.
    """
    # 2.1 Definition: Sigmoid Activation Function using standard formula 1 / (1 + e^-z).
    # 2.2 Why: Maps input to a probability range (0 to 1). Crucial for binary classification.
    # 2.3 When: Used as the activation output of the final layer.
    # 2.4 Where: Last step of forward pass.
    # 2.5 How to use: `s = 1 / (1 + np.exp(-z))`.
    # 2.6 How it works: Exponentiates the negative input, adds 1, and inverts. Large pos -> 1, Large neg -> 0.
    # 2.7 Output: Float value between 0 and 1.
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    3.1 Argument `z`: The Z value (linear output) from the forward pass.
    Note: Ideally we use 'a' (sigmoid output) for efficiency: a * (1 - a).
    Here we recompute sigmoid(z) for clarity.
    """
    # 2.1 Definition: Derivative of Sigmoid function.
    # 2.2 Why: Required for Backpropagation to calculate gradients (Chain Rule).
    # 2.3 When: During the Backward pass (Output layer derivative).
    # 2.4 Where: Calculating `dZ2`.
    # 2.5 How to use: `grad = sigmoid_derivative(z)`.
    # 2.6 How it works: `s * (1 - s)`. Steepest at z=0 (grad=0.25), flat at extremes.
    # 2.7 Output: Float, derivative value.
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)


def relu(z):
    """
    3.1 Argument `z`: Input array.
    """
    # 2.1 Definition: Rectified Linear Unit (ReLU).
    # 2.2 Why: Introduces non-linearity without the vanishing gradient problem of sigmoid for deep layers.
    # 2.3 When: Hidden layers activation.
    # 2.4 Where: Between Linear Step 1 and Linear Step 2.
    # 2.5 How to use: `np.maximum(0, z)`.
    # 2.6 How it works: Returns z if z > 0, else 0.
    # 2.7 Output: Array with negative values replaced by 0.
    return np.maximum(0, z)


def relu_derivative(z):
    """
    3.1 Argument `z`: Input array from forward pass.
    """
    # 2.1 Definition: Derivative of ReLU.
    # 2.2 Why: To pass gradients through the ReLU activation during backprop.
    # 2.3 When: Calculating `dZ1` (Hidden layer error).
    # 2.4 Where: Backward pass step.
    # 2.5 How to use: `gradient = (z > 0).astype(float)`.
    # 2.6 How it works: Gradient is 1 if z > 0, else 0. Undefined at z=0 (practically 0).
    # 2.7 Output: Binary mask (0s and 1s).
    dZ = np.array(z, copy=True)
    dZ[z <= 0] = 0
    dZ[z > 0] = 1
    return dZ


# =============================================================================
# DATASET CREATION
# =============================================================================

# 2.1 Definition: Define Input Features (X).
# 2.2 Why: The XOR problem consists of 4 possible inputs.
# 2.3 When: Before training starts.
# 2.4 Where: Global scope.
# 2.5 How to use: Numpy array of shape (4, 2).
# 2.6 How it works: Creates a matrix.
# 2.7 Output: [[0,0], [0,1], [1,0], [1,1]].
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 2.1 Definition: Define True Labels (Y).
# 2.2 Why: Supervised learning requires targets to calculate error. XOR logic: 0 if same, 1 if different.
# 2.3 When: Before training.
# 2.4 Where: Global scope.
# 2.5 How to use: Numpy array of shape (4, 1). Reshaped to column vector.
# 2.6 How it works: Creates a label vector.
# 2.7 Output: [[0], [1], [1], [0]].
Y = np.array([[0], [1], [1], [0]])


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_parameters():
    """
    No arguments. Returns initialized dictionary of parameters.
    """
    # 2.1 Definition: Initialize Weights W1 (Input -> Hidden).
    # 2.2 Why: Connects 2 input neurons to 2 hidden neurons.
    # 2.3 When: Start of training.
    # 2.4 Where: Parameter dictionary.
    # 2.5 How to use: `randn(2, 2) * 0.5`.
    # 2.6 How it works: Small random numbers break symmetry. Factor 0.5 reduces variance.
    # 2.7 Output: 2x2 Matrix.
    W1 = np.random.randn(2, 2) * 0.5
    
    # 2.1 Definition: Initialize Bias b1 (Hidden).
    # 2.2 Why: Allows shifting the activation function.
    # 2.3 When: Start of training.
    # 2.4 Where: Parameter dictionary.
    # 2.5 How to use: `zeros((1, 2))`.
    # 2.6 How it works: Start neutral.
    # 2.7 Output: [0, 0].
    b1 = np.zeros((1, 2))
    
    # 2.1 Definition: Initialize Weights W2 (Hidden -> Output).
    # 2.2 Why: Connects 2 hidden neurons to 1 output neuron.
    # 2.3 When: Start of training.
    # 2.4 Where: Parameter dictionary.
    # 2.5 How to use: `randn(2, 1) * 0.5`.
    # 2.6 How it works: Random initialization.
    # 2.7 Output: 2x1 Matrix.
    W2 = np.random.randn(2, 1) * 0.5
    
    # 2.1 Definition: Initialize Bias b2 (Output).
    # 2.2 Why: Shift output activation.
    # 2.3 When: Start of training.
    # 2.4 Where: Parameter dictionary.
    # 2.5 How to use: `zeros((1, 1))`.
    # 2.6 How it works: Start neutral.
    # 2.7 Output: [0].
    b2 = np.zeros((1, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

def forward_propagation(X, parameters):
    """
    3.1 Argument `X`: Input data matrix (Batch Size, Input Features).
    3.2 Argument `parameters`: Dictionary containing W1, b1, W2, b2.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 2.1 Definition: Linear Step 1 (Z1).
    # 2.2 Why: Combine inputs with weights.
    # 2.3 When: Layer 1 calculation.
    # 2.4 Where: Forward pass (Hidden Layer).
    # 2.5 How to use: `dot(X, W1) + b1`.
    # 2.6 How it works: Matrix multiplication adds weighted inputs. Bias shifts result.
    # 2.7 Output: Matrix (Batch, Hidden Nodes).
    Z1 = np.dot(X, W1) + b1
    
    # 2.1 Definition: Activation Step 1 (A1).
    # 2.2 Why: Apply non-linearity (ReLU).
    # 2.3 When: After Z1.
    # 2.4 Where: Hidden Layer Output.
    # 2.5 How to use: `relu(Z1)`.
    # 2.6 How it works: Zeros out negatives.
    # 2.7 Output: Matrix (Batch, Hidden Nodes) >= 0.
    A1 = relu(Z1)
    
    # 2.1 Definition: Linear Step 2 (Z2).
    # 2.2 Why: Combine hidden features for final decision.
    # 2.3 When: Output Layer calculation.
    # 2.4 Where: Output Layer.
    # 2.5 How to use: `dot(A1, W2) + b2`.
    # 2.6 How it works: Weighted sum of activated features.
    # 2.7 Output: Matrix (Batch, Output Nodes).
    Z2 = np.dot(A1, W2) + b2
    
    # 2.1 Definition: Activation Step 2 (A2).
    # 2.2 Why: Convert score to probability (Sigmoid).
    # 2.3 When: Final Output.
    # 2.4 Where: Output Layer.
    # 2.5 How to use: `sigmoid(Z2)`.
    # 2.6 How it works: S-curve squash.
    # 2.7 Output: Probabilities (0 to 1).
    A2 = sigmoid(Z2)
    
    # Store values for backprop
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def compute_loss(A2, Y):
    """
    3.1 Argument `A2`: Predicted probabilities.
    3.2 Argument `Y`: True labels.
    """
    m = Y.shape[0] # Number of samples
    
    # 2.1 Definition: Binary Cross-Entropy Loss (Log Loss).
    # 2.2 Why: Standard loss for binary classification. Penalizes wrong confident predictions highly.
    # 2.3 When: Evaluating model performance after forward pass.
    # 2.4 Where: Training loop.
    # 2.5 How to use: `-1/m * sum(Y*log(P) + (1-Y)*log(1-P))`.
    # 2.6 How it works: If Y=1, maximizes log(P). If Y=0, maximizes log(1-P).
    # 2.7 Output: Scalar float (Loss value).
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    loss = -1/m * np.sum(logprobs)
    return np.squeeze(loss) # Ensure scalar


# =============================================================================
# BACKWARD PROPAGATION
# =============================================================================

def backward_propagation(parameters, cache, X, Y):
    """
    3.1 Argument `parameters`: Weights need for shape references or derivative logic.
    3.2 Argument `cache`: Stored Z and A values from forward pass.
    3.3 Argument `X`, `Y`: Input and Truth.
    """
    m = X.shape[0]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    
    # 2.1 Definition: Calculate dZ2 (Output Error).
    # 2.2 Why: Gradient of Loss wrt Z2 (Output Layer Linear).
    # 2.3 When: First step of backprop (last layer).
    # 2.4 Where: Output Layer.
    # 2.5 How to use: `A2 - Y`.
    # 2.6 How it works: For Cross-Entropy + Sigmoid, derivative simplifies to Pred - truth.
    # 2.7 Output: Error matrix (Batch, 1).
    dZ2 = A2 - Y
    
    # 2.1 Definition: Calculate dW2 (Gradient of W2).
    # 2.2 Why: To know how to adjust W2 to reduce loss.
    # 2.3 When: Updating Output Layer weights.
    # 2.4 Where: Output Layer.
    # 2.5 How to use: `1/m * dot(A1.T, dZ2)`.
    # 2.6 How it works: Average contribution of each hidden node to the error.
    # 2.7 Output: Gradient Matrix same shape as W2.
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    
    # 2.1 Definition: Calculate db2 (Gradient of b2).
    # 2.2 Why: To adjust bias.
    # 2.3 When: Updating Output Layer bias.
    # 2.4 Where: Output Layer.
    # 2.5 How to use: `1/m * sum(dZ2)`.
    # 2.6 How it works: Average error across batch.
    # 2.7 Output: Scalar/Vector same shape as b2.
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    # 2.1 Definition: Calculate dA1 (Hidden Layer Error contribution).
    # 2.2 Why: Propagate error backwards to hidden layer.
    # 2.3 When: Transition from Output to Hidden.
    # 2.4 Where: Between layers.
    # 2.5 How to use: `dot(dZ2, W2.T)`.
    # 2.6 How it works: Weighted sum of output errors back to hidden nodes.
    # 2.7 Output: Error matrix for A1.
    dA1 = np.dot(dZ2, W2.T)
    
    # 2.1 Definition: Calculate dZ1 (Hidden Layer Linear Error).
    # 2.2 Why: Account for ReLU activation derivative.
    # 2.3 When: Before calculating W1/b1 gradients.
    # 2.4 Where: Hidden Layer.
    # 2.5 How to use: `dA1 * relu_derivative(Z1)`.
    # 2.6 How it works: If input was <= 0, gradient is killed (0), else passes through.
    # 2.7 Output: Error matrix for Z1.
    dZ1 = dA1 * relu_derivative(Z1)
    
    # 2.1 Definition: Calculate dW1 (Gradient of W1).
    # 2.2 Why: To update Input weights.
    # 2.3 When: Final step of backprop.
    # 2.4 Where: Input Layer.
    # 2.5 How to use: `1/m * dot(X.T, dZ1)`.
    # 2.6 How it works: Average contribution of inputs to hidden error.
    # 2.7 Output: Gradient Matrix same shape as W1.
    dW1 = (1/m) * np.dot(X.T, dZ1)
    
    # 2.1 Definition: Calculate db1 (Gradient of b1).
    # 2.2 Why: To update Hidden bias.
    # 2.3 When: Final step of backprop.
    # 2.4 Where: Input Layer.
    # 2.5 How to use: `1/m * sum(dZ1)`.
    # 2.6 How it works: Average error.
    # 2.7 Output: Vector same shape as b1.
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=0.1):
    """
    3.1 Argument `grads`: Gradients calculated from backprop.
    3.2 Argument `learning_rate`: Step size for update.
    """
    # 2.1 Definition: Parameter Update Rule.
    # 2.2 Why: Moving W in opposite direction of gradient reduces loss.
    # 2.3 When: After Backprop.
    # 2.4 Where: Optimization step.
    # 2.5 How to use: `W = W - alpha * dW`.
    # 2.6 How it works: Gradient Descent.
    # 2.7 Output: Updated Dictionary.
    
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    
    return parameters


# =============================================================================
# GRADIENT CHECKING
# =============================================================================

def gradient_check(parameters, X, Y, epsilon=1e-4):
    """
    3.1 Argument `epsilon`: Small perturbation value (finite difference).
    """
    print("\n--- Running Gradient Check ---")
    
    # 1. Compute Analytical Gradient via Backprop
    A2, cache = forward_propagation(X, parameters)
    grads = backward_propagation(parameters, cache, X, Y)
    grad_analytical = grads["dW1"][0, 0] # Check specific weight W1[0,0]
    
    # 2. Compute Numerical Gradient via Finite Differences
    
    # Save original value
    W1_original = parameters["W1"][0, 0]
    
    # Perturb + epsilon
    parameters["W1"][0, 0] = W1_original + epsilon
    A2_plus, _ = forward_propagation(X, parameters)
    loss_plus = compute_loss(A2_plus, Y)
    
    # Perturb - epsilon
    parameters["W1"][0, 0] = W1_original - epsilon
    A2_minus, _ = forward_propagation(X, parameters)
    loss_minus = compute_loss(A2_minus, Y)
    
    # Restore original
    parameters["W1"][0, 0] = W1_original
    
    # Calculate Numerical Gradient: (J+ - J-) / (2*eps)
    grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
    
    # 3. Calculate Difference
    diff = abs(grad_analytical - grad_numerical)
    
    print(f"Analytical Gradient (dW1[0,0]): {grad_analytical:.8f}")
    print(f"Numerical Gradient  (dW1[0,0]): {grad_numerical:.8f}")
    print(f"Absolute Difference: {diff:.8f}")
    
    if diff < 1e-3:
        print(">> Gradient Check PASSED!")
    else:
        print(">> Gradient Check FAILED!")
        
    return diff


# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

# 2.1 Definition: Initialize Model Parameters.
parameters = initialize_parameters()

# 2.1 Definition: Run Gradient Check before training.
# 2.2 Why: To verify correctness of backpropagation implementation.
gradient_check(parameters, X, Y, epsilon=1e-4)

print("\n--- Starting Training ---")
iterations = 5000

for i in range(iterations):
    # 1. Forward
    A2, cache = forward_propagation(X, parameters)
    
    # 2. Loss
    cost = compute_loss(A2, Y)
    
    # 3. Backward
    grads = backward_propagation(parameters, cache, X, Y)
    
    # 4. Update
    parameters = update_parameters(parameters, grads, learning_rate=0.1)
    
    # 2.1 Definition: Log Progress.
    # 2.5 How to use: Modulo operator to print every 500 steps.
    if i % 500 == 0:
        print(f"Iteration {i}: Loss = {cost:.5f}")


# =============================================================================
# FINAL RESULTS
# =============================================================================

print("\n--- Final Results ---")
print(f"Iteration {iterations}: Loss = {cost:.5f}")

# 2.1 Definition: Generate Final Predictions.
A2_final, _ = forward_propagation(X, parameters)

print("\nTruth Table Comparison:")
print("Input | Label | Prediction | Prob | Correct?")
print("-" * 45)

for j in range(len(X)):
    input_val = X[j]
    true_label = Y[j][0]
    prob = A2_final[j][0]
    pred_label = 1 if prob > 0.5 else 0
    is_correct = "Yes" if pred_label == true_label else "No"
    
    print(f"{input_val} |   {true_label}   |     {pred_label}      | {prob:.2f} | {is_correct}")

