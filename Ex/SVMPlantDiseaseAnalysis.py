"""
Problem Statement:
SVM Kernel Selection and Hyperparameter Tuning for Plant Disease Classification.
Context:
- Dataset: 15,000 images, 10 disease classes, 512 CNN features.
- Kernel Performance:
  * Linear: 73% accuracy, 80ms latency
  * Polynomial (degree=2): 79% accuracy, 120ms latency
  * Polynomial (degree=3): 82% accuracy, 180ms latency
  * RBF: 85% accuracy, 150ms latency
- Requirements: >80% accuracy AND <200ms latency.

Sub-Problems:
1. Part A: Select the best kernel (RBF) with 4 technical justifications.
2. Part B: Explain C parameter role. Prefer large or small C for this problem?
3. Part C: Explain gamma parameter. Systematic tuning approach for C and gamma.

Steps to Solve:
1. Simulate CNN features (512-dimensional data, 10 classes).
2. Train SVM with different kernels and measure performance.
3. Demonstrate Grid Search for C and gamma tuning.
4. Print detailed answers for Parts A, B, and C.

Expected Output:
- Performance comparison table (Accuracy vs Latency).
- Grid Search results showing optimal C and gamma.
- Text answers explaining kernel selection, C parameter implications, and tuning strategy.
"""

"""
Task: Part A (Kernel Selection)
----------------------------------------
Question: Which kernel would you choose? Provide four technical justifications.
Answer: I would choose the RBF (Radial Basis Function) kernel.

Justification 1 (Accuracy Requirement):
RBF achieves 85% accuracy, which exceeds the >80% threshold. Linear (73%) and Poly-2 (79%) fail this requirement.
Poly-3 meets accuracy (82%) but is close to the boundary, leaving no safety margin.

Justification 2 (Latency Constraint):
RBF's 150ms prediction time is well within the <200ms requirement with 50ms buffer.
Poly-3 at 180ms is risky - any production overhead could push it over the limit.

Justification 3 (Feature Space Characteristics):
CNN features are high-dimensional (512D) and non-linearly separable. RBF maps to infinite dimensions,
capturing complex disease patterns (texture, color variations) that linear boundaries cannot.
The 12% accuracy gap (Linear 73% vs RBF 85%) proves strong non-linearity exists.

Justification 4 (Scalability and Robustness):
RBF has only 2 hyperparameters (C, gamma) vs Polynomial's 3 (C, gamma, degree).
Simpler tuning space means faster optimization and more stable production performance.
Additionally, RBF is less prone to numerical instability than high-degree polynomials.
"""

"""
Task: Part B (C Parameter)
----------------------------------------
Question: Explain C parameter role. Prefer large or small C? Discuss FP/FN implications.
Answer:

C Parameter Role:
C controls the regularization strength (inverse). It balances:
- Margin Maximization (generalization): Prefer wider decision boundaries.
- Training Error Minimization: Correctly classify training points.

Large C: Hard margin. Forces model to classify every training point correctly.
- Risk: Overfitting. Model memorizes noise, poor generalization.
Small C: Soft margin. Allows some misclassifications for smoother boundaries.
- Risk: Underfitting. May miss genuine disease patterns.

Recommendation for Plant Disease: Moderately Large C (e.g., C=10).

Why:
False Negative (Missing Disease): CRITICAL. A farmer who doesn't treat a diseased plant loses the crop.
False Positive (Misdiagnosis): TOLERABLE. Farmer applies wrong treatment, but at least investigates.

Moderately large C ensures we capture subtle disease features (high training accuracy) while
maintaining some regularization to avoid overfitting to image artifacts (lighting, background noise).
We prioritize Recall (catching diseases) over Precision (avoiding false alarms).
"""

"""
Task: Part C (Gamma Parameter and Tuning)
----------------------------------------
Question: Explain gamma. Systematic approach to tune C and gamma. Validation strategy?
Answer:

Gamma Parameter:
Gamma controls the "reach" of each training example's influence.
- High gamma: Only nearby points affect decision boundary (complex, wiggly boundaries).
  Risk: Overfitting to individual images.
- Low gamma: Distant points influence boundary (smoother, simpler boundaries).
  Risk: Underfitting, missing local disease clusters.

Systematic Tuning Approach:
1. Grid Search with Cross-Validation:
   - C range: [0.1, 1, 10, 100] (logarithmic scale)
   - Gamma range: [0.001, 0.01, 0.1, 1] (logarithmic scale)
   - Total: 16 combinations

2. Validation Strategy: Stratified 5-Fold Cross-Validation
   Why Stratified: Ensures each fold has balanced disease classes (10 classes).
   Why 5-Fold: Balances computational cost (15K images) with reliable estimates.

3. Evaluation Metric: F1-Score (macro-averaged)
   Why: Handles class imbalance better than accuracy. Equally weights all 10 diseases.

4. Final Validation: Hold-out test set (20% of data) for unbiased performance estimate.

5. Latency Check: Measure prediction time on test set to confirm <200ms constraint.
"""

# Why: Import NumPy for efficient numerical array handling (feature matrices).
# Output: Module 'numpy' loaded as alias 'np'.
import numpy as np

# Why: Import Pandas for tabular data formatting (Comparison Table).
# Output: Module 'pandas' loaded as alias 'pd'.
import pandas as pd

# Why: Import Time to measure inference latency (Milliseconds).
# Output: Module 'time' loaded.
import time

# Why: Import 'make_classification' to generate synthetic dataset matching the problem specs (15k images, 512 features).
# Output: Function loaded.
from sklearn.datasets import make_classification

# Why: Import 'train_test_split' to create a withheld test set for unbiased validation.
# Why: Import 'GridSearchCV' to systematically find the best C and Gamma.
# Why: Import 'StratifiedKFold' to ensure all 10 disease classes are represented in every CV fold.
# Output: Selection tools loaded.
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Why: Import 'SVC' (Support Vector Classifier), the core algorithm for this task.
# Output: Class loaded.
from sklearn.svm import SVC

# Why: Import metrics. 'accuracy_score' for the >80% req, 'f1_score' for class imbalance, 'classification_report' for details.
# Output: Metric functions loaded.
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ==========================================
# 1. Data Simulation (CNN Features)
# ==========================================

print("--- 1. Data Simulation (Plant Disease Dataset) ---")

# Generate Data
# What: Create a synthetic dataset that mimics the complexity of 512-feature CNN embeddings.
# When: At the start of the script to provide data for the pipeline.
np.random.seed(42)
X, y = make_classification(
    # n_samples=15000: 
    # What: Total number of "images" to generate.
    # Why: Matches the problem statement's dataset size.
    n_samples=15000,
    
    # n_features=512:
    # What: Dimensionality of each sample.
    # Why: Simulates the "512 CNN-extracted features" mentioned.
    n_features=512,
    
    # n_informative=400:
    # What: Number of features that actually carry signal about the disease class.
    # Why: High number chooses to simulate rich, complex CNN embeddings.
    n_informative=400,
    
    # n_redundant=50:
    # What: Features that are linear combinations of informative ones.
    # Why: Adds realistic noise/redundancy often found in raw data.
    n_redundant=50,
    
    # n_classes=10:
    # What: Number of distinct targets (0-9).
    # Why: Matches the "10 disease classes" requirement.
    n_classes=10,
    
    # n_clusters_per_class=3:
    # What: Number of distinct "types" of images within a single disease class.
    # Why: Simulates variations (e.g., Early blight vs Late blight appearance) within the same label.
    n_clusters_per_class=3, 
    
    # random_state=42:
    # What: Seed for reproducibility.
    # Why: Ensures we get the exact same dataset every time we run.
    random_state=42
)
# Output (X): Matrix of shape (15000, 512) containing float values.
# Output (y): Array of shape (15000,) containing integers 0-9.

print(f"Dataset Shape: {X.shape}")
print(f"Class Distribution: {np.bincount(y)}")

# Split Data
# What: Divides the dataset into Training set (for learning) and Test set (for independent evaluation).
# Output: 4 arrays: X_train (12000 images), X_test (3000 images), y_train, y_test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    
    # test_size=0.2:
    # What: 20% of data goes to the Test set.
    # Why: Standard ratio. 3000 images is statistically large enough for a reliable accuracy estimate.
    test_size=0.2, 
    
    # random_state=42:
    # What: Feature shuffling seed.
    # Why: Ensures the specific images selected for 'Test' are the same every run.
    random_state=42, 
    
    # stratify=y:
    # What: Ensures the Test set has the exact same class distribution (e.g., 10% of Class 0, 10% of Class 1) as the original data.
    # Why: Crucial for Classification. Prevents a "lucky" test set that has no hard classes.
    stratify=y
)
# Output (X_train): (12000, 512) Feature matrix.
# Output (X_test): (3000, 512) Feature matrix.
# Output (y_train): (12000,) Label vector.
# Output (y_test): (3000,) Label vector.

# ==========================================
# 2. Kernel Comparison (Part A Support)
# ==========================================

print("\n--- 2. Kernel Performance Comparison ---")

# Define Kernels
# What: List of kernel configurations to test.
# Why: To empirically validate the problem statement's reported performance.
kernels = [
    {'name': 'Linear', 'kernel': 'linear'},
    {'name': 'Poly-2', 'kernel': 'poly', 'degree': 2},
    {'name': 'Poly-3', 'kernel': 'poly', 'degree': 3},
    {'name': 'RBF', 'kernel': 'rbf'}
]

results = []

for config in kernels:
    # What: Extract the specific display name (e.g., 'Poly-2') from the dictionary.
    name = config.pop('name')
    
    # SVC Initialization
    # What: Create an instance of the Support Vector Classifier.
    # When: At the start of each loop iteration.
    # Why: We need a fresh model for each kernel type.
    # Arguments:
    # - **config: Unpacks the dict (e.g., kernel='poly', degree=2).
    # - random_state=42: Used for probability estimates (if enabled) and internal shuffling.
    svm = SVC(**config, random_state=42)
    
    # Fit (Training)
    # What: The heavy lifting. Solves the Quadratic Programming problem to find the optimal hyperplane.
    # When: After initialization.
    # Why: This learns the decision boundary from the 12,000 training images.
    svm.fit(X_train, y_train)
    
    # Latency Measurement Setup
    # What: Record the current system timestamp.
    # Why: To calculate the duration of the prediction phase accurately.
    start = time.time()
    
    # Prediction (Inference)
    # What: Pass 3000 test images through the learned boundary to get labels.
    # When: During validation.
    y_pred = svm.predict(X_test)
    
    # Calculate Latency
    # What: (End Time - Start Time) * 1000.
    # Why: Converts seconds to milliseconds (requirements are in ms).
    latency = (time.time() - start) * 1000
    
    # Accuracy Calculation
    # What: Counts how many predictions match the ground truth (y_test).
    # Output: Float (e.g., 0.85).
    acc = accuracy_score(y_test, y_pred)
    
    # Store Result
    # What: Append a dictionary of metrics for this kernel to the results list.
    results.append({
        'Kernel': name,
        'Accuracy': f"{acc:.2%}",
        'Latency (ms)': f"{latency:.1f}",
        'Meets Accuracy (>80%)': '✓' if acc > 0.80 else '✗',
        'Meets Latency (<200ms)': '✓' if latency < 200 else '✗'
    })

# Display Results
# What: Pretty-print comparison table.
# Output: DataFrame showing all metrics.
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ==========================================
# 3. Hyperparameter Tuning (Part C)
# ==========================================

print("\n--- 3. Grid Search for C and Gamma (RBF Kernel) ---")

# Define Parameter Grid
# What: Logarithmic ranges for C and gamma.
# Why: These parameters span orders of magnitude, so log scale is appropriate.
# Output: Dictionary of parameter lists.
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Grid Search with Cross-Validation
# What: Exhaustive search over parameter grid with 5-fold CV.
# When: During hyperparameter optimization phase.
# Why: 
#   - cv=StratifiedKFold ensures balanced classes in each fold.
#   - scoring='f1_macro' handles multi-class imbalance better than accuracy.
#   - n_jobs=-1 uses all CPU cores for parallel processing.
# Output: GridSearchCV object with best parameters.
# CV Strategy
# What: Define the Cross-Validation splitting method.
# Why: Standard KFold might split classes unevenly. StratifiedKFold maintains the 10% per class ratio in every fold.
cv_strategy = StratifiedKFold(
    # n_splits=5:
    # What: Divide data into 5 chunks.
    # Why: 5 is the industry standard (Pareto principle).
    n_splits=5,
    
    # shuffle=True:
    # What: Randomize data ordering before splitting.
    # Why: Disperses any inherent ordering artifacts (e.g., if data was sorted by timestamp).
    shuffle=True,
    
    # random_state=42:
    # What: Fix the random seed.
    # Why: Reproducibility.
    random_state=42
)

# Initialize GridSearchCV
# What: The wrapper that automates the loop: "For every C in List, For every Gamma in List, Do CV".
# When: Before training.
grid_search = GridSearchCV(
    # estimator=SVC(...):
    # What: The machine learning algorithm to optimize. RBF kernel is hardcoded here based on Part A results.
    estimator=SVC(kernel='rbf', random_state=42),
    
    # param_grid=param_grid:
    # What: The dictionary of parameters to test ({'C': [...], 'gamma': [...]}).
    param_grid=param_grid,
    
    # cv=cv_strategy:
    # What: The splitting object defined above (Stratified 5-Fold).
    cv=cv_strategy,
    
    # scoring='f1_macro':
    # What: The metric to maximize. 'macro' calculates F1 for each of 10 classes and averages them equally.
    # Why: 'accuracy' is biased if one class is huge. 'f1_macro' ensures we care about small disease classes too.
    scoring='f1_macro',
    
    # n_jobs=-1:
    # What: Use all available CPU cores.
    # Why: To run the 80 models in parallel, drastically reducing wait time.
    n_jobs=-1,
    
    # verbose=1:
    # What: Print progress updates to console.
    verbose=1
)

# Fit Grid Search
# What: Triggers the exhaustive search.
# 1. Generates 16 combinations (C x Gamma).
# 2. For each combo, runs 5-Fold CV (Train on 4 folds, Validate on 1).
# 3. Averages the scores.
# 4. Refits the best combo on the entire X_train.
# Output: A populated grid_search object containing the best model.
print("Running Grid Search (this may take a minute)...")
grid_search.fit(X_train, y_train)

# Best Parameters
# What: Extract optimal C and gamma values.
# Output: Dictionary of best hyperparameters.
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# Final Test Set Evaluation
# What: Validate performance on the 3000 images we hid at the very beginning.
# Why: This simulates "Real World" performance on brand new data the model has never seen.

# Retrieve Best Model
# What: Get the SVC object configured with the winning C and Gamma (e.g., C=10, gamma=0.01).
# Output: SVC model object.
best_model = grid_search.best_estimator_

# Predict on Test Set
# What: Generate labels (0-9) for the test images Use the optimized model.
# Output: Array of 3000 integers.
y_pred_final = best_model.predict(X_test)

# Calculate Macro F1
# What: Compute F1 score for each class separately and take the unweighted average.
# Why: Ensures that "Disease 9" (rare) is treated as equally important as "Disease 0" (common).
test_f1 = f1_score(y_test, y_pred_final, average='macro')

# Calculate Accuracy
# What: Standard percent correct.
test_acc = accuracy_score(y_test, y_pred_final)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_acc:.2%}")
print(f"  F1-Score (macro): {test_f1:.4f}")

# Latency Check for Best Model
# Why: The optimal C/Gamma might act differently than default params. We must re-verify speed.

# Start Timer
# What: Capture current clock time.
start = time.time()

# Dummy Prediction
# What: Run prediction purely to measure time. We ignore the output variable ('_').
_ = best_model.predict(X_test)

# Calculate duration
# What: (Stop - Start) * 1000.
# Output: Latency in milliseconds.
final_latency = (time.time() - start) * 1000

print(f"  Latency: {final_latency:.1f} ms")
print(f"  Meets <200ms requirement: {'✓' if final_latency < 200 else '✗'}")

# ==========================================
# 4. Detailed Classification Report
# ==========================================

print("\n--- 4. Per-Class Performance (Best Model) ---")

# Classification Report
# What: Generates a text summary of the main classification metrics.
# Arguments:
# - y_test: Ground truth.
# - y_pred_final: Model predictions.
# - target_names: List of strings to label rows (Disease_0, etc.) instead of just numbers.
# Output: Formatted string table.
print(classification_report(y_test, y_pred_final, target_names=[f"Disease_{i}" for i in range(10)]))
