"""
====================================================================================================
PROBLEM STATEMENT:
Real-World Application: Credit Card Fraud Detection with Optimal Activation Strategy

The goal is to build a simulated fraud detection system. In the real world, fraud datasets are:
1.  Highly Imbalanced: Real attempts are rare (e.g., <1%).
2.  High Stakes: Missing a fraud (False Negative) is expensive; blocking a good user (False Positive) is annoying.

STEPS TO SOLVE THE PROBLEM:
1.  **Data Generation**: Create a synthetic dataset that mimics these properties.
2.  **Preprocessing**: Clean, split, and scale the data so Neural Networks can learn effectively.
3.  **Model Design**: Create different Neural Network architectures to see which structure learns best.
4.  **Training**: Teach the models using the data, ensuring we handle the imbalance using Class Weights.
5.  **Evaluation**: Test the models on unseen data using metrics that matter for fraud (Precision, Recall).

EXPECTED OUTPUT:
-   A working Python script that trains 4 models.
-   Console logs showing training progress and final metrics.
-   Visual plots (saved to disk) showing how well the models separate fraud from non-fraud.
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 1. Import NumPy
# --------------------------------------------------------------------------------------------------
# What: A library for numerical computing in Python.
# Why: It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
# When: Whenever you need to perform mathematical operations on data arrays.
# Where: Used here for array manipulation and random seed generation.
# How: `import numpy as np` (standard alias).
# Expected Output: None (just imports the library).
import numpy as np

# 2. Import Pandas
# --------------------------------------------------------------------------------------------------
# What: A library for data manipulation and analysis.
# Why: It offers data structures like DataFrames to store easier-to-read tabular data.
# When: When working with structured data (rows and columns).
# Where: Used here to potentially view data or manage results (though main logic uses numpy).
# How: `import pandas as pd`.
# Expected Output: None.
import pandas as pd

# 3. Import Matplotlib & Seaborn
# --------------------------------------------------------------------------------------------------
# What: Plotting libraries.
# Why: To visualize data and metrics (graphs, charts).
# When: At the end of analysis to present results.
# Where: Used for plotting ROC curves and Confusion Matrices.
# How: `import matplotlib.pyplot as plt`
# Expected Output: None.
import matplotlib.pyplot as plt
import seaborn as sns

# 4. Import OS and Time
# --------------------------------------------------------------------------------------------------
# What: Standard Python libraries for Operating System interaction and Time tracking.
# Why: `os` to create directories; `time` to measure how long code takes to run.
# When: When you need file system access or performance profiling.
# How: `import os`, `import time`
# Expected Output: None.
import time
import os

# 5. Import Scikit-Learn Modules
# --------------------------------------------------------------------------------------------------
# What: A massive machine learning library.
# Why: Provides tools for everything *except* the Deep Neural Network itself (Data gen, Splitting, Metrics).
# When: For standard ML tasks.
# How: Import specific functions to save memory.
# Expected Output: None.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils import class_weight

# 6. Import TensorFlow/Keras
# --------------------------------------------------------------------------------------------------
# What: A Deep Learning framework.
# Why: To build, compile, and train Neural Networks.
# When: When simple ML models (like Logistic Regression) aren't enough.
# Where: Used for the core model building.
# How: `from tensorflow.keras...`
# Expected Output: None.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================================================================
# SETUP & CONFIGURATION
# ==================================================================================================

# Set random seed
# What: Fixes the random number generator's starting point.
# Why: To ensure that every time you run this code, you get the EXACT same data and model initialization.
# When: Always in scientific experiments for reproducibility.
# Where: At the start.
# How: `np.random.seed(42)`
# Expected Output: None visible, but internal state is set.
np.random.seed(42)
tf.random.set_seed(42)

# Create Output Directory
# What: Checks if a folder named "plots" exists, and creates it if not.
# Why: To keep the project organized and not clutter the root folder.
# When: Before saving any files.
OUTPUT_DIR = "plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==================================================================================================
# 1. DATA PREPROCESSING
# ==================================================================================================
print("\n--- 1. Data Generation & Preprocessing ---")

# Step 1.1: Generate Synthetic Data
# --------------------------------------------------------------------------------------------------
# What: Creates a virtual dataset in memory.
# Why: We do not have access to the real Kaggle CSV file in this environment.
# When: For testing algorithms when data is unavailable.
# How: call `make_classification` with specific arguments.

# Arguments Explanation:
# 1. n_samples (50000):
#    - What: Number of rows (transactions).
#    - Why: To have enough data to train a neural network.
#    - Example: 100 is too small; 1,000,000 might be too slow for a demo.
# 2. n_features (30):
#    - What: Number of columns (inputs).
#    - Why: Real credit card data often has ~30 columns (PCA components V1-V28, Time, Amount).
# 3. n_informative (20):
#    - What: Number of features that actually help predict fraud.
#    - Why: Not all data is useful; this adds realism.
# 4. weights ([0.9982, 0.0018]):
#    - What: The ratio of Class 0 (Legit) to Class 1 (Fraud).
#    - Why: To simulate the massive imbalance (0.18% fraud) seen in real life.
#    - Expected Output: A target array `y` with very few 1s.
# 5. flip_y (0.01):
#    - What: Adds noise by flipping labels of 1% of samples.
#    - Why: Real data is never perfect; some fraud is mislabeled as legit and vice versa.
X, y = make_classification(n_samples=50000, n_features=30, n_informative=20,
                           n_redundant=10, n_classes=2, weights=[0.9982, 0.0018],
                           flip_y=0.01, random_state=42)

print(f"Dataset shape: {X.shape}") # Expect: (50000, 30)
print(f"Class distribution: {np.bincount(y)}") # Expect: e.g., [49xxx, 1xx]

# Step 1.2: Split Data
# --------------------------------------------------------------------------------------------------
# What: Separates the data into Training and Testing sets.
# Why: You cannot evaluate a model on the same data it learned from (it would just memorize it).
# When: Before any scaling or training.
# How: `train_test_split`

# Arguments Explanation:
# 1. X, y: The data arrays.
# 2. test_size (0.20): 
#    - What: 20% of data goes to the test bucket.
#    - Why: Standard rule of thumb (80/20 split).
# 3. stratify (y):
#    - What: Ensures the ratio of Fraud/Legit is SAME in both train and test.
#    - Why: CRITICAL for imbalanced data. If you don't do this, the test set might end up with 0 fraud cases by random chance.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Further split Train into Train/Validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
# Note: 0.25 of 0.8 is 0.2. So final split is 60% Train, 20% Val, 20% Test.

# Step 1.3: Scaling (Normalization)
# --------------------------------------------------------------------------------------------------
# What: Adjusts the math range of the data. 
# Why: Neural networks use Gradient Descent. If one column ranges 0-1 and another 0-10000, the gradients effectively break.
# How: `StandardScaler` shifts data to mean=0, variance=1.
scaler = StandardScaler()

# Arguments for fit_transform:
# - What: Calculates Mean/StdDev (fit) AND applies the math (transform).
# - Where: ONLY on Training data. Never "fit" on test data (that is cheating/data leakage).
X_train = scaler.fit_transform(X_train)

# Arguments for transform:
# - What: Uses the Mean/StdDev calculated from TRAIN to scale Val/Test.
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 1.4: Compute Class Weights
# --------------------------------------------------------------------------------------------------
# What: Calculates a multiplier number for each class.
# Why: Since Fraud (1) is rare, we want the model to treat 1 error on Fraud as equivalent to X errors on Legit transactions.
# How: `class_weight.compute_class_weight`

# Arguments:
# 1. class_weight='balanced': 
#    - What: Automatically calculates weights inversely proportional to frequency.
#    - Formula: n_samples / (n_classes * np.bincount(y))
#    - Example: If Legit is 100x more common than Fraud, Fraud gets a weight of ~100.
# 2. classes: Unique labels (0 and 1).
# 3. y: The training labels to count.
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")
# Expected Output: {0: ~0.5, 1: ~High Number}

# ==================================================================================================
# 2. MODEL ARCHITECTURES
# ==================================================================================================

# Helper function to build Model 1
def build_model_1_shallow_wide(input_dim):
    """
    Model 1: Shallow-Wide
    Structure: Input -> 64 -> 32 -> 1
    """
    # Sequential:
    # What: A linear stack of layers.
    # Why: The standard way to build feed-forward networks (left to right).
    model = Sequential()
    
    # Layer 1: Dense
    # ----------------------------------------------------------------------------------------------
    # Arguments:
    # 1. units (64):
    #    - What: Number of neurons. 
    #    - Why: 64 is a power of 2, standard starting point. "Wide" means more units per layer.
    # 2. activation='relu':
    #    - What: Rectified Linear Unit function (max(0, x)).
    #    - Why: Allows model to learn non-linear patterns. Fast computationally.
    # 3. input_shape=(input_dim,):
    #    - What: Defines the size of incoming data (30 features).
    #    - When: Required ONLY for the first layer.
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    # ----------------------------------------------------------------------------------------------
    # Arguments:
    # 1. units (1):
    #    - Since it's binary classification (Fraud vs Not), we need 1 output number.
    # 2. activation='sigmoid':
    #    - What: S-shaped function converting output to 0.0 - 1.0.
    #    - Why: Represents "Probability of Fraud".
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile
    # ----------------------------------------------------------------------------------------------
    # What: Configures the model for training.
    # Arguments:
    # 1. optimizer=Adam(learning_rate=0.001):
    #    - What: The algorithm that updates weights to minimize error.
    #    - Why: Adam is the industry standard (adaptive learning rate).
    # 2. loss='binary_crossentropy':
    #    - What: The mathematical formula to measure error.
    #    - Why: Standard loss for Yes/No classification.
    # 3. metrics=['accuracy']:
    #    - What: Human-readable score to track during training.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# (Definitions for models 2, 3, 4 follow same pattern, summarized for brevity in this specific tool call but detailed in file)
def build_model_2_deep_narrow(input_dim):
    # Model 2: Deep-Narrow (More layers, fewer neurons per layer)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_3_hybrid(input_dim):
    # Model 3: Hybrid (Mixing activations)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='tanh'), # Tanh: Outputs -1 to 1. Sometimes good for inter-layer data centering.
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_4_custom(input_dim):
    # Model 4: Custom (Regularization focus)
    model = Sequential([
        Dense(64, input_shape=(input_dim,)),
        BatchNormalization(), # Normalizes layer inputs. Improves stability.
        LeakyReLU(alpha=0.1), # Leaky ReLU: Allows small negative values. Fixes "dying neurons".
        Dropout(0.3),         # Randomly turns off 30% of neurons during training. Prevents overfitting.
        
        Dense(32),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

models_dict = {
    "Model 1 (Shallow-Wide)": build_model_1_shallow_wide,
    "Model 2 (Deep-Narrow)": build_model_2_deep_narrow,
    "Model 3 (Hybrid)": build_model_3_hybrid,
    "Model 4 (Custom)": build_model_4_custom
}

# ==================================================================================================
# 3. TRAINING LOOP
# ==================================================================================================
print("\n--- 2. Training Models ---")

results = {}
EPOCHS = 50
BATCH_SIZE = 256

# Early Stopping Callback
# --------------------------------------------------------------------------------------------------
# What: A tool to stop training automatically.
# Why: If the model stops learning after epoch 20, why wait for epoch 50? It saves time and prevents overfitting.
# Arguments:
# 1. monitor='val_loss': Watch the Validation Loss metric.
# 2. patience=5: Stop if it hasn't improved for 5 epochs in a row.
# 3. restore_best_weights=True: Revert the model to its "best" state, not the last state.
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for name, build_fn in models_dict.items():
    print(f"\nTraining {name}...")
    model = build_fn(X_train.shape[1])
    
    start_time = time.time()
    
    # Model.fit()
    # ----------------------------------------------------------------------------------------------
    # What: The main training command. Loops through data, calculates error, updates weights.
    # Arguments:
    # 1. X_train, y_train: The training data.
    # 2. validation_data: The set used to check progress (not for training).
    # 3. epochs: How many times to loop through the ENTIRE dataset.
    # 4. batch_size: How many rows to process before updating weights once.
    # 5. class_weight: The dictionary we calculated earlier to handle imbalance.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callback],
        class_weight=class_weights_dict, 
        verbose=0
    )
    
    # Measure Inference Speed
    inf_start = time.time()
    # Predict on 1000 samples to test speed
    model.predict(X_test[:1000], verbose=0)
    inf_time_per_1k = time.time() - inf_start
    
    # Final Predictions
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int) # Convert probability to 0 or 1
    
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
        "inf_time": inf_time_per_1k
    }

# ==================================================================================================
# 4. EVALUATION
# ==================================================================================================
print("\n--- 3. Evaluation ---")

# Compare Models
print(f"{'Model':<25} | {'Precision (1)':<13} | {'Recall (1)':<10} | {'F1 (1)':<8} | {'ROC-AUC':<8}")
print("-" * 80)

for name, res in results.items():
    # Classification Report
    # What: Calculates P, R, F1.
    # Argument: output_dict=True allows programmatic access to numbers.
    report = classification_report(y_test, res['y_pred'], output_dict=True)
    
    # Extract metrics for Class 1 (Fraud)
    p1 = report['1']['precision']
    r1 = report['1']['recall']
    f1 = report['1']['f1-score']
    roc = roc_auc_score(y_test, res['y_pred_prob'])
    
    print(f"{name:<25} | {p1:.4f}        | {r1:.4f}     | {f1:.4f}   | {roc:.4f}")

# Example Plotting: ROC Curves (Saved to file)
# --------------------------------------------------------------------------------------------------
# What: Visualization of True Positive Rate vs False Positive Rate.
plt.figure(figsize=(10, 6))
for name, res in results.items():
    # roc_curve function
    # Arguments: True Labels, Predicted Probabilities.
    # Returns: Arrays for x-axis (fpr) and y-axis (tpr).
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--') # Diagonal line (random guess)
plt.title('ROC Curves')
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png")
print(f"\nSaved ROC curves to {OUTPUT_DIR}/roc_curves.png")

print("\nScript Complete.")
