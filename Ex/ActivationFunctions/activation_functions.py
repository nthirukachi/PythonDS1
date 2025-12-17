"""
# Problem Statement:
# Compare Activation Functions Mathematically and Visually [CODING]
# Implement and analyze the three major activation functions: Sigmoid, Tanh, and ReLU.
#
# Steps to solve the problem:
# 1. Implement Activation Functions and Derivatives:
#    - Define `sigmoid(z)` and `sigmoid_derivative(z)`.
#    - Define `tanh(z)` and `tanh_derivative(z)`.
#    - Define `relu(z)` and `relu_derivative(z)`.
# 2. Visualization:
#    - Generate input data range [-6, 6].
#    - Plot all three functions on one graph.
#    - Plot all three derivatives on a separate graph.
#    - Create a side-by-side comparison figure.
# 3. Numerical Analysis:
#    - Calculate outputs for specific inputs [-5, -2, -0.5, 0, 0.5, 2, 5].
#    - Identify ranges where gradients are strong (> 0.1).
#    - Display gradient values at key points (-2, 0, 2).
# 4. Analysis & Observations:
#    - Document the vanishing gradient problem, saturation regions, and preferred use cases.
#
# Expected Output:
# - Three saved plot images.
# - Console output with a comparison table and gradient analysis.
"""

import numpy as np # Import numpy for numerical operations and array handling
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs

# --- 1. Define Activation Functions & Derivatives ---

# What: Sigmoid Activation Function
# Why: Maps any input value to a value between 0 and 1. Used for probability output (binary classification).
# When: Typically in the output layer of binary classifiers.
# Formula: 1 / (1 + e^-z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# What: Derivative of Sigmoid
# Why: Used during backpropagation to calculate gradients.
# When: Updating weights in a network using Sigmoid.
# Formula: sigmoid(z) * (1 - sigmoid(z))
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# What: Tanh (Hyperbolic Tangent) Activation Function
# Why: Maps input to [-1, 1]. Zero-centered, which usually helps convergence compared to Sigmoid.
# When: Hidden layers of neural networks.
# Formula: (e^z - e^-z) / (e^z + e^-z)
def tanh(z):
    return np.tanh(z)

# What: Derivative of Tanh
# Why: Used for backpropagation.
# When: Updating weights in a network using Tanh.
# Formula: 1 - tanh(z)^2
def tanh_derivative(z):
    t = tanh(z)
    return 1 - t**2

# What: ReLU (Rectified Linear Unit) Activation Function
# Why: Outputs input if positive, else 0. Solves vanishing gradient prob for positive inputs. Computationally efficient.
# When: Most hidden layers in modern deep learning models (CNNs, etc.).
# Formula: max(0, z)
def relu(z):
    return np.maximum(0, z)

# What: Derivative of ReLU
# Why: Backpropagation. Gradient is 1 for z > 0, 0 for z < 0.
# When: Updating weights in ReLU networks.
# Formula: 1 if z > 0 else 0
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# --- Analysis Driver Code ---

# Set up input range for visualization
# What: Create an array of values from -6 to 6
# Why: To visualize the behavior of functions over a standard range covering linear and saturation regions.
z_values = np.linspace(-6, 6, 400)

# Calculate function outputs
sig_out = sigmoid(z_values)
tanh_out = tanh(z_values)
relu_out = relu(z_values)

# Calculate derivative outputs
sig_deriv = sigmoid_derivative(z_values)
tanh_deriv = tanh_derivative(z_values)
relu_deriv = relu_derivative(z_values)

# --- 2. Visualizations ---

# Plot 1: Activation Functions
plt.figure(figsize=(10, 6))
plt.plot(z_values, sig_out, label='Sigmoid', color='blue')
plt.plot(z_values, tanh_out, label='Tanh', color='red')
plt.plot(z_values, relu_out, label='ReLU', color='green')
plt.title('Activation Functions Comparison')
plt.xlabel('Input (z)')
plt.ylabel('Output (Activation)')
plt.ylim(-1.5, 2) # Limit y-axis to focus on relevant range
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5) # Add x-axis line
plt.axvline(0, color='black', linewidth=0.5) # Add y-axis line
plt.savefig('c:/nagpython/demouv/Ex/ActivationFunctions/activations_plot.png')
print("Saved activations_plot.png")

# Plot 2: Derivatives
plt.figure(figsize=(10, 6))
plt.plot(z_values, sig_deriv, label="Sigmoid'", color='blue')
plt.plot(z_values, tanh_deriv, label="Tanh'", color='red')
plt.plot(z_values, relu_deriv, label="ReLU'", color='green')
plt.title('Derivatives of Activation Functions')
plt.xlabel('Input (z)')
plt.ylabel('Gradient Value')
plt.ylim(-0.2, 1.2)
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.savefig('c:/nagpython/demouv/Ex/ActivationFunctions/derivatives_plot.png')
print("Saved derivatives_plot.png")

# Plot 3: Side-by-Side Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Functions
ax1.plot(z_values, sig_out, label='Sigmoid', color='blue')
ax1.plot(z_values, tanh_out, label='Tanh', color='red')
ax1.plot(z_values, relu_out, label='ReLU', color='green')
ax1.set_title('Activation Functions')
ax1.set_ylim(-1.5, 2)
ax1.grid(True)
ax1.legend()

# Subplot 2: Derivatives
ax2.plot(z_values, sig_deriv, label="Sigmoid'", color='blue')
ax2.plot(z_values, tanh_deriv, label="Tanh'", color='red')
ax2.plot(z_values, relu_deriv, label="ReLU'", color='green')
ax2.set_title('Derivatives')
ax2.set_ylim(-0.2, 1.2)
ax2.grid(True)
ax2.legend()

plt.savefig('c:/nagpython/demouv/Ex/ActivationFunctions/comparison_plot.png')
print("Saved comparison_plot.png")


# --- 3. Numerical Analysis ---

test_inputs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])

print("\n--- Numerical Analysis Table ---")
print(f"{'Input':<10} | {'Sigmoid':<10} | {'Tanh':<10} | {'ReLU':<10}")
print("-" * 46)
for z in test_inputs:
    s = sigmoid(z)
    t = tanh(z)
    r = relu(z)
    print(f"{z:<10} | {s:<10.4f} | {t:<10.4f} | {r:<10.4f}")

print("\n--- Gradient Analysis Table ---")
print(f"{'Input':<10} | {'Sigmoid\'':<10} | {'Tanh\'':<10} | {'ReLU\'':<10}")
print("-" * 46)
for z in test_inputs:
    sd = sigmoid_derivative(z)
    td = tanh_derivative(z)
    rd = relu_derivative(z)
    print(f"{z:<10} | {sd:<10.4f} | {td:<10.4f} | {rd:<10.4f}")

# Identify strong gradient regions (> 0.1)
print("\n--- Strong Gradient Regions (> 0.1) Analysis ---")
# Check a subset of calculations
points_of_interest = [-2, 0, 2]
for p in points_of_interest:
    print(f"At x = {p}:")
    if sigmoid_derivative(p) > 0.1: print(f"  - Sigmoid has strong gradient ({sigmoid_derivative(p):.4f})")
    else: print(f"  - Sigmoid has weak gradient ({sigmoid_derivative(p):.4f}) -> Vanishing risk")
    
    if tanh_derivative(p) > 0.1: print(f"  - Tanh has strong gradient ({tanh_derivative(p):.4f})")
    else: print(f"  - Tanh has weak gradient ({tanh_derivative(p):.4f}) -> Vanishing risk")
    
    if relu_derivative(p) > 0.1: print(f"  - ReLU has strong gradient ({relu_derivative(p):.4f})")
    else: print(f"  - ReLU has weak/zero gradient ({relu_derivative(p):.4f})")

# --- 4. Written Analysis (as comments) ---
"""
Written Analysis:

1. Vanishing Gradient Problem:
   - Sigmoid: Note how the derivative is close to 0 for inputs < -2 and > 2. This is the vanishing gradient problem. 
     Deep networks struggle to learn because updates become tiny. Max gradient is only 0.25 at z=0.
   - Tanh: Similar issue, saturates at < -2 and > 2, but max gradient is 1.0 at z=0, so it's slightly better than Sigmoid.
   - ReLU: Derivative is exactly 1 for all z > 0. It does NOT vanish for positive inputs, making it ideal for deep networks. 
     However, for z < 0, gradient is 0 ("Dead ReLU" problem).

2. Saturation Regions:
   - Sigmoid: Saturates (output flat) when input is very positive (-> 1) or very negative (-> 0).
   - Tanh: Saturates (output flat) when input is very positive (-> 1) or very negative (-> -1).
   - ReLU: Does not saturate in the positive direction.

3. Preferences:
   - Use Sigmoid: Only for binary classification output layers.
   - Use Tanh: Sometimes in hidden layers if inputs are normalized, or in RNNs/LSTMs.
   - Use ReLU: Default for hidden layers in almost all modern neural networks (CNNs, MLPs) due to speed and gradient properties.
"""
