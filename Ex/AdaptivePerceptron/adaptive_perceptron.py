"""
====================================================================================================
1. PROBLEM STATEMENT:
Build an adaptive perceptron for a data stream with mild concept drift using a Functional Approach.

We need to:
1.  Generate a data stream using `drifting_stream` (3 batches).
2.  Implement Perceptron logic using **independent functions** (no class):
    -   `predict`: Calculate output.
    -   `train_step`: Update weights.
    -   `adapt`: Handle decay and reset logic.
3.  Stream processing:
    -   Maintain state (weights, bias, lr) in the main loop.
    -   Validate on buffer (200 samples).
    -   Check for drift (Acc < 0.70) -> Reset state.
    -   Decay learning rate every 5 epochs.
4.  Visualization: Plot Accuracy vs Batch index.

STEPS TO SOLVE THE PROBLEM:
1.  Data Generation: `drifting_stream` generates (X, y) batches.
2.  Functional Definitions:
    -   `initialize_weights(n_features)`: Returns random weights and 0 bias.
    -   `predict(X, weights, bias)`: Returns labels.
    -   `train_step(X, y, weights, bias, lr)`: Returns updated (weights, bias).
    -   `adapt(accuracy, epoch_idx, lr, initial_lr)`: Returns new (lr, reset_flag).
3.  Main Loop:
    -   Initialize state variables.
    -   Loop through batches.
    -   Predict validation -> Calc Accuracy.
    -   Call `adapt` -> Update LR or Reset weights if needed.
    -   Call `train_step` -> Update weights/bias.
4.  Output: Metrics and Plot.

EXPECTED OUTPUT:
-   A plot showing accuracy dips at drift points and recovery.
-   Console logs showing LR decay and "RESET TRIGGERED" messages.
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports numpy.
# 2.2: Why it is used: For array math (dot products, addition).
# 2.3: When to used: Linear algebra ops.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `import numpy as np`.
# 2.6: How it works: Loads C library.
# 2.7: Output: Module.
import numpy as np

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports matplotlib.
# 2.2: Why it is used: Plotting.
# 2.6: How it works: Visualization backend.
# 2.7: Output: Module.
import matplotlib.pyplot as plt

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports data generator.
# 2.2: Why it is used: Synthetic data.
# 2.6: How it works: Sklearn function.
# 2.7: Output: Function.
from sklearn.datasets import make_classification

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports accuracy metric.
# 2.2: Why it is used: Performance tracking.
# 2.6: How it works: Comparison.
# 2.7: Output: Function.
from sklearn.metrics import accuracy_score

# ==================================================================================================
# 1. DATA GENERATION
# ==================================================================================================
# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates drifting stream data.
# 2.2: Why it is used: To create the problem setup.
# 2.6: How it works: Returns list of 3 batches.
# 2.7: Output: List.
def drifting_stream(seed=99):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Init RNG.
    # 3. Arguments:
    # 3.1 seed: Random seed (e.g. 99).
    rng = np.random.default_rng(seed)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Init list.
    # 2.7: Output: [].
    batches = []
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Define shifts.
    # 2.7: Output: List of tuples.
    shifts = [(0.0, 0.0), (0.8, -0.6), (1.2, 0.9)]
    
    for drift_x, drift_y in shifts:
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Create batch.
        # 2.7: Output: X, y.
        X, y = make_classification(
            n_samples=500, n_features=2, n_informative=2, n_redundant=0,
            class_sep=1.2, random_state=rng.integers(1000)
        )
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Apply drift.
        # 2.7: Output: Modified array.
        X[:, 0] += drift_x
        X[:, 1] += drift_y
        
        batches.append((X, y))
    
    return batches

# ==================================================================================================
# 2. FUNCTIONAL PERCEPTRON LOGIC
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes weights and bias.
# 2.2: Why it is used: Need a starting point for the model.
# 2.6: How it works: Random small numbers for weights, 0 for bias.
# 2.7: Output: Tuple (weights, bias).
# 3. Arguments:
#    3.1 n_features: Number of input columns.
#    3.6 Example: 2
def initialize_weights(n_features):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Generates random weights.
    # 2.6: How it works: randn returns standard normal.
    # 2.7: Output: Array.
    weights = np.random.randn(n_features) * 0.01
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Init bias.
    # 2.7: Output: 0.
    bias = 0
    return weights, bias

# 2. Detailed Explanation:
# 2.1: What the line of code does: Predicts labels given inputs and state.
# 2.2: Why it is used: Inference.
# 2.6: How it works: Dot product + threshold.
# 2.7: Output: Binary array.
# 3. Arguments:
#    3.1 X: Data.
#    3.6 Example: X_val
#    3.1 weights: Weight vector.
#    3.6 Example: np.array([0.1, -0.2])
#    3.1 bias: Bias scalar.
#    3.6 Example: 0.05
def predict(X, weights, bias):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Calculates linear score.
    # 2.6: How it works: Math.
    # 2.7: Output: Vector.
    linear_output = np.dot(X, weights) + bias
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Applies step function.
    # 2.6: How it works: Elementwise comparison.
    # 2.7: Output: Int array.
    return np.where(linear_output >= 0, 1, 0)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Performs one epoch of training (updates weights).
# 2.2: Why it is used: Learning.
# 2.6: How it works: Iterates samples, applies perceptron rule.
# 2.7: Output: Tuple (new_weights, new_bias).
# 3. Arguments:
#    3.1 X: Training Data.
#    3.6 Example: X_train
#    3.1 y: Training Labels.
#    3.6 Example: y_train
#    3.1 weights: Current weights.
#    3.6 Example: w
#    3.1 bias: Current bias.
#    3.6 Example: b
#    3.1 lr: Current learning rate.
#    3.6 Example: 0.001
def train_step(X, y, weights, bias, lr):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Iterates batch.
    # 2.7: Output: Iterator.
    for i, x_i in enumerate(X):
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Predict single sample.
        # 2.6: How it works: Reshapes to (1, n) for dot prod.
        # 2.7: Output: 0 or 1.
        y_pred = predict(x_i.reshape(1, -1), weights, bias)[0]
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Calculates error term.
        # 2.7: Output: Scalar.
        update = lr * (y[i] - y_pred)
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Update weights in place (for efficiency, but we return them too).
        # 2.7: Output: None.
        weights += update * x_i
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Update bias.
        # 2.7: Output: None.
        bias += update
        
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Return new state.
    # 2.7: Output: Tuple.
    return weights, bias

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adapts LR or signals Reset based on performance.
# 2.2: Why it is used: Concept drift handling.
# 2.6: How it works: Logic check (acc < 0.70).
# 2.7: Output: Tuple (new_lr, reset_triggered_bool).
# 3. Arguments:
#    3.1 accuracy: Current score.
#    3.6 Example: 0.65
#    3.1 epoch_idx: Batch number.
#    3.6 Example: 2
#    3.1 lr: Current LR.
#    3.6 Example: 0.009
#    3.1 initial_lr: Baseline LR.
#    3.6 Example: 0.1
def adapt(accuracy, epoch_idx, lr, initial_lr):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Check for drift.
    # 2.7: Output: Boolean.
    if accuracy < 0.70:
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Return initial LR and True for reset.
        # 2.2: Why it is used: Restart learning.
        # 2.7: Output: Tuple.
        return initial_lr, True
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Check for decay schedule.
    # 2.7: Output: Boolean.
    elif epoch_idx > 0 and epoch_idx % 5 == 0:
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Return decayed LR and False.
        # 2.7: Output: Tuple.
        return lr * 0.90, False
        
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Return unchanged LR.
    # 2.7: Output: Tuple.
    return lr, False

# ==================================================================================================
# 3. MAIN STREAMING LOOP
# ==================================================================================================
print("\n--- Starting Functional Adaptive Perceptron Stream ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Load data.
# 2.7: Output: List.
batches = drifting_stream()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Define Initial Hyperparameters.
# 2.7: Output: Scalar.
INITIAL_LR = 0.1

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initialize State Variables.
# 2.2: Why it is used: Functional approach needs external state management.
# 2.7: Output: Scalar/Array.
current_lr = INITIAL_LR
weights, bias = initialize_weights(2) # 2 features

# 2. Detailed Explanation:
# 2.1: What the line of code does: History tracking.
# 2.7: Output: Lists.
acc_history = []
reset_events = []

# 2. Detailed Explanation:
# 2.1: What the line of code does: Loop over batches.
# 2.7: Output: Iterator.
for i, (X_batch, y_batch) in enumerate(batches):
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Split Val/Train.
    # 2.7: Output: Arrays.
    X_val, y_val = X_batch[:200], y_batch[:200]
    X_train, y_train = X_batch[200:], y_batch[200:]
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Predict and Score.
    # 2.7: Output: Float.
    y_pred_val = predict(X_val, weights, bias)
    acc = accuracy_score(y_val, y_pred_val)
    acc_history.append(acc)
    
    print(f"Batch {i}: Accuracy = {acc:.2f} | LR = {current_lr:.4f}")
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Call adapt logic.
    # 2.2: Why it is used: Get new LR and check if reset needed.
    # 2.7: Output: Tuple.
    new_lr, should_reset = adapt(acc, i, current_lr, INITIAL_LR)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Update global current_lr.
    # 2.7: Output: Assignment.
    current_lr = new_lr
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Handle reset if signaled.
    # 2.7: Output: Boolean check.
    if should_reset:
        print(f"   >>> RESET TRIGGERED! Weights cleared.")
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Re-initialize weights.
        # 2.2: Why it is used: Wipe memory.
        # 2.7: Output: New state.
        weights, bias = initialize_weights(2)
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Record event.
        # 2.7: Output: List append.
        reset_events.append(i)
        
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Train on current batch.
    # 2.2: Why it is used: Update weights.
    # 2.7: Output: New state tuple.
    weights, bias = train_step(X_train, y_train, weights, bias, current_lr)

# ==================================================================================================
# 4. VISUALIZATION
# ==================================================================================================
print("\n--- Generating Visualization ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Set up plot.
# 2.7: Output: Figure.
plt.figure(figsize=(10, 5))

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plot accuracy.
# 2.7: Output: Line.
plt.plot(range(len(acc_history)), acc_history, marker='o', label='Validation Accuracy')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plot resets.
# 2.7: Output: Vertical lines.
for r in reset_events:
    plt.axvline(x=r, color='red', linestyle='--', label='Weight Reset')

plt.axhline(y=0.70, color='gray', linestyle=':', label='Threshold (0.70)')
plt.xlabel('Batch Index')
plt.ylabel('Accuracy')
plt.title('Functional Adaptive Perceptron Performance under Drift')
plt.legend()
plt.grid(True)
plt.savefig('drift_analysis.png')
print("Plot saved to drift_analysis.png")
