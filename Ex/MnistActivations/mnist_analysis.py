"""
# Problem Statement:
# Build a Multi-Class Image Classifier with Activation Function Analysis [CODING]
# Build a neural network to classify handwritten digits (MNIST) and analyze how activation function choice
# impacts performance, training time, and gradient flow.
#
# Steps to solve the problem:
# 1. Data Preparation:
#    - Load MNIST dataset.
#    - Preprocess data: Flatten images (28x28 -> 784) and normalize pixel values (0-255 -> 0-1).
#    - One-hot encode targets? Keras 'sparse_categorical_crossentropy' handles integers, so not strictly needed but good practice. We'll use sparse for simplicity.
# 2. Model Architecture:
#    - Define a function `build_model` that creates a Keras Sequential model.
#    - Layers: Dense(128) -> Dense(64) -> Dense(10, softmax).
#    - The activation function for hidden layers will be a parameter (sigmoid, tanh, relu).
# 3. Training Loop:
#    - Iterate through the 3 activation types.
#    - Compile model with Adam optimizer (lr=0.001).
#    - Train for 20 epochs, recording history (loss/acc) and time per epoch.
# 4. Gradient Analysis:
#    - After training, use `tf.GradientTape` to compute the gradient of the loss with respect to the *first layer's weights* for a fixed batch.
#    - This measures how much signal reaches the start of the network (checking for vanishing gradients).
# 5. Visualization:
#    - Plot Accuracy vs Epochs for all models.
#    - Plot Loss vs Epochs.
#    - Bar chart for Final Test Accuracy.
#    - Bar chart for Average Training Time.
#    - Bar chart for Gradient Magnitudes.
#
# Expected Output:
# - Console output with training progress and final metrics.
# - 5 Image files containing the plots.
"""

import tensorflow as tf # Import TensorFlow library for Deep Learning
from tensorflow.keras import layers, models, optimizers # Import Keras modules for building models
from tensorflow.keras.datasets import mnist # Import MNIST dataset handler
import numpy as np # Import NumPy for array operations
import matplotlib.pyplot as plt # Import Matplotlib for plotting
import time # Import time module to track training duration

# What: Set random seed
# Why: To ensure results are reproducible (weights initialize identically).
# Example: Running the script twice yields the same curves.
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Data Loading and Preprocessing ---

# What: Load data
# Why: Get the standard handwritten digit dataset.
# Output: Tuple of Numpy arrays.
print("Loading MNIST data...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# What: Preprocess inputs
# Why: Neural networks converge faster on small, normalized values (0 to 1) than large integers (0 to 255). Flattening is needed for Dense layers.
# Steps: Reshape to (Samples, 784), Divide by 255.0.
# Output shape: (60000, 784)
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# --- 2. Model Building Function ---

# What: Define a model builder
# Why: We need to create 3 almost identical models, changing only the activation. A function avoids code duplication.
# When: Called inside the experiment loop.
# Arguments: activation_name (e.g., 'relu', 'tanh', 'sigmoid')
def build_model(activation_name):
    model = models.Sequential()
    # Layer 1: Input to Hidden 1 (128 neurons)
    # What: Dense (Fully Connected) Layer
    # Why: Learn patterns from input pixels.
    # input_shape=(784,) defines the expected input vector size.
    model.add(layers.Dense(128, activation=activation_name, input_shape=(784,)))
    
    # Layer 2: Hidden 1 to Hidden 2 (64 neurons)
    # What: Second Dense Layer
    # Why: Learn more complex abstract features.
    model.add(layers.Dense(64, activation=activation_name))
    
    # Output Layer: Hidden 2 to Output (10 neurons)
    # What: Final classification layer.
    # Why: We have 10 classes (digits 0-9). 'softmax' converts outputs to probabilities summing to 1.
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# --- 3. Experiment Setup ---

activations = ['sigmoid', 'tanh', 'relu']
results = {} # Dictionary to store history and metrics for each model

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 4. Training Loop ---

for act in activations:
    print(f"\n--- Training Model with {act.upper()} activation ---")
    
    # What: Create model instance
    model = build_model(act)
    
    # What: Compile model
    # Why: Configure the training process.
    # Optimizer: Adam is efficient and standard.
    # Loss: Sparse Categorical Crossentropy acts on integer labels directly.
    # Metrics: We want to monitor accuracy.
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # What: Train model & Measure time
    start_time = time.time()
    
    # What: Fit the model
    # Output: 'history' object containing loss/acc per epoch.
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    end_time = time.time()
    avg_time_per_epoch = (end_time - start_time) / EPOCHS
    
    # --- 5. Gradient Analysis ---
    # What: Calculate gradient of loss w.r.t first layer weights
    # Why: To check for vanishing gradients. If gradients are ~0, the first layer isn't learning.
    # When: After training (to see the state of the trained model) or typically during training. 
    # Here we check it on a batch to see magnitude.
    
    # Get a small batch for gradient calculation
    X_batch = X_train[:32]
    y_batch = y_train[:32]
    
    # Use GradientTape to separate execution for gradient recording
    with tf.GradientTape() as tape:
        # Forward pass on batch
        predictions = model(X_batch, training=False)
        # Compute loss
        loss_val = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        
    # Calculate gradients w.r.t TRAINABLE VARIABLES
    # We are interested in the First Layer's Weights (model.trainable_variables[0])
    # Usually: [Layer1_Weights, Layer1_Biases, Layer2_Weights...]
    grads = tape.gradient(loss_val, model.trainable_variables)
    
    # Layer 0 Weights are usually at index 0.
    # shape: (784, 128)
    first_layer_grads = grads[0]
    
    # Mean Absolute Gradient: Average strength of the update signal
    mean_abs_grad = np.mean(np.abs(first_layer_grads.numpy()))
    
    print(f"Mean Absolute Gradient (Layer 1): {mean_abs_grad:.6f}")
    
    # Store results
    results[act] = {
        'history': history.history,
        'test_acc': history.history['val_accuracy'][-1],
        'avg_time': avg_time_per_epoch,
        'grad_magnitude': mean_abs_grad
    }

# --- 6. Visualization ---

# Plot 1: Accuracy Comparison
plt.figure(figsize=(10, 6))
for act in activations:
    plt.plot(results[act]['history']['accuracy'], label=f'{act} (Train)')
    plt.plot(results[act]['history']['val_accuracy'], linestyle='--', label=f'{act} (Val)')
plt.title('Model Accuracy: Train vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('c:/nagpython/demouv/Ex/MnistActivations/accuracy_plot.png')
print("Saved accuracy_plot.png")

# Plot 2: Loss Comparison
plt.figure(figsize=(10, 6))
for act in activations:
    plt.plot(results[act]['history']['loss'], label=f'{act} (Train)')
    plt.plot(results[act]['history']['val_loss'], linestyle='--', label=f'{act} (Val)')
plt.title('Model Loss: Train vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('c:/nagpython/demouv/Ex/MnistActivations/loss_plot.png')
print("Saved loss_plot.png")

# Plot 3: Final Test Accuracy Bar Chart
plt.figure(figsize=(8, 5))
accs = [results[act]['test_acc'] for act in activations]
plt.bar(activations, accs, color=['blue', 'orange', 'green'])
plt.title('Final Validation Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0) # Zoom in for clarity
plt.grid(axis='y')
plt.savefig('c:/nagpython/demouv/Ex/MnistActivations/bar_accuracy.png')
print("Saved bar_accuracy.png")

# Plot 4: Training Time Bar Chart
plt.figure(figsize=(8, 5))
times = [results[act]['avg_time'] for act in activations]
plt.bar(activations, times, color=['blue', 'orange', 'green'])
plt.title('Average Training Time per Epoch (seconds)')
plt.ylabel('Time (s)')
plt.grid(axis='y')
plt.savefig('c:/nagpython/demouv/Ex/MnistActivations/bar_time.png')
print("Saved bar_time.png")

# Plot 5: Gradient Magnitude Bar Chart
plt.figure(figsize=(8, 5))
grads = [results[act]['grad_magnitude'] for act in activations]
plt.bar(activations, grads, color=['blue', 'orange', 'green'])
plt.title('First Layer Gradient Magnitude (Post-Training)')
plt.ylabel('Mean Abs Gradient')
plt.yscale('log') # Use log scale because differences can be huge (vanishing gradient)
plt.grid(axis='y')
plt.savefig('c:/nagpython/demouv/Ex/MnistActivations/bar_gradient.png')
print("Saved bar_gradient.png")

# --- Summary Table to Console ---
print("\n--- Summary Metrics ---")
print(f"{'Activation':<10} | {'Test Acc':<10} | {'Time/Epoch':<10} | {'Gradient Mag':<15}")
print("-" * 55)
for act in activations:
    r = results[act]
    print(f"{act:<10} | {r['test_acc']:.4f}     | {r['avg_time']:.4f}s    | {r['grad_magnitude']:.6f}")
