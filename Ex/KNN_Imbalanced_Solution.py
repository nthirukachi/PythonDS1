"""
Problem Statement:
Imbalanced Classification using k-NN with Preprocessing.
Scenario:
- Dataset: 1,000 patients, 50 features (normalized 0-1).
- Target: Rare disease (Binary: 1 = Disease, 0 = Healthy).
- Class Balance: 50 Positives (5%), 950 Negatives (95%).
- Objective: Apply k-NN with k ~ 5% of N, handling imbalance and dimensionality.

Preprocessing Steps Implemented:
1. Creating the Imbalanced Dataset.
2. Handling Imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   - Why: To prevent k-NN from being biased toward the majority class (Healthy).
3. Dimensionality Reduction using PCA (Principal Component Analysis).
   - Why: To mitigate the 'Curse of Dimensionality' and improve distance metric reliability.
4. Tuning k:
   Calculation:
   - N (Training Data) = 1,000 (approx)
   - Requirement: 5% of N
   - k = 1000 * 0.05 = 50.
   - Selected k = 51 (Nearest odd number to avoid ties).
"""

# Why: Import numpy for efficient array handling and numerical operations.
# Output: Module 'numpy' loaded as 'np'.
import numpy as np

# Why: Import pandas to handle data in a tabular format (though mostly using numpy arrays here).
# Output: Module 'pandas' loaded as 'pd'.
import pandas as pd

# Why: Import PCA to reduce the dimensionality of the 50-feature dataset.
# Output: Class 'PCA' imported.
from sklearn.decomposition import PCA

# Why: Import KNeighborsClassifier to perform the classification.
# Output: Class 'KNeighborsClassifier' imported.
from sklearn.neighbors import KNeighborsClassifier

# Why: Import train_test_split to separate training and evaluation data.
# Output: Function 'train_test_split' imported.
from sklearn.model_selection import train_test_split

# Why: Import SMOTE to generate synthetic samples for the minority class (rare disease).
# Note: This requires 'imbalanced-learn' library (`pip install imbalanced-learn`).
# Output: Class 'SMOTE' imported.
from imblearn.over_sampling import SMOTE

# Why: Import metrics to evaluate the model (Confusion Matrix, Classification Report).
# Output: Function 'classification_report' imported.
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# Step 1: Generate Synthetic Dataset
# ==========================================

# Why: Seed for reproducibility so we get the same random numbers every time.
# Output: Random state set to 42.
np.random.seed(42)

# Why: Define dataset parameters based on the problem statement.
n_samples = 1000
n_features = 50

# Why: Generate 1000 patients with 50 random features (already normalized 0-1).
# np.random.rand returns values from [0, 1).
# Output: X shape is (1000, 50).
X = np.random.rand(n_samples, n_features)

# Why: Create the target labels. First 950 are 0 (Healthy), last 50 are 1 (Disease).
# np.zeros(950): Vector of 950 zeros.
# np.ones(50): Vector of 50 ones.
# np.hstack: Concatenates them horizontally into one array.
# Output: y shape is (1000,) with 950 '0's and 50 '1's.
y = np.hstack((np.zeros(950), np.ones(50)))

# Why: Shuffle the data because currently all 1s are at the end.
# permutation: Returns a scrambled range of indices [0, 1, ..., 999].
indices = np.random.permutation(n_samples)
# Output: X and y are now shuffled in the same order.
X, y = X[indices], y[indices]

print(f"Original Class Distribution: {np.bincount(y.astype(int))}")
# Output Example: Original Class Distribution: [950  50]

# ==========================================
# Step 2: Split Data (BEFORE Preprocessing)
# ==========================================

# Why: Split into Train and Test. We must split BEFORE applying SMOTE to avoid data leakage.
# (We don't want synthetic test data derived from training data to leak into the validation).
# Output: X_train (800, 50), X_test (200, 50).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# Step 3: Dimensionality Reduction (PCA)
# ==========================================

# Why: Reduce features from 50 to a smaller number (e.g., 15) to remove noise and improve k-NN distance metric.
# n_components=15: Arbitrary choice, usually determined by "explained variance ratio".
# Output: PCA object initialized.
pca = PCA(n_components=15)

# Why: Fit PCA on training data to learn the principal components.
# transform: Apply the reduction to X_train.
# Output: X_train_pca now has shape (800, 15).
X_train_pca = pca.fit_transform(X_train)

# Why: Transform test data using the SAME PCA components learned from training.
# Output: X_test_pca now has shape (200, 15).
X_test_pca = pca.transform(X_test)

# ==========================================
# Step 4: Handle Class Imbalance (SMOTE)
# ==========================================

# Why: The training set still has ~5% positives. k-NN will likely ignore them.
# SMOTE generates new synthetic points between existing positive samples.
# Output: SMOTE object initialized.
smote = SMOTE(random_state=42)

# Why: Resample X_train_pca and y_train.
# Output: X_train_resampled will have balanced classes (e.g., 760 vs 760).
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

print(f"Resampled Class Distribution: {np.bincount(y_train_resampled.astype(int))}")
# Output Example: Resampled Class Distribution: [760 760] (Assuming 80% split of 950 negatives)

# ==========================================
# Step 5: k-NN Classification
# ==========================================

# Part A Calculation:
# Total Training Samples (N after SMOTE) is roughly 1520.
# But original problem asked for k based on 5% of training data (N=1000 -> k=50).
# Let's stick to the problem logic: k = 50.
# We choose k=51 to avoid ties (odd number).

# Why: Initialize k-NN classifier with k=51.
# metric='minkowski', p=2: This is standard Euclidean distance.
# Output: Classifier object created.
k = 51
knn = KNeighborsClassifier(n_neighbors=k)

# Why: Train the k-NN model on the BALANCED (SMOTE) and REDUCED (PCA) training set.
# Output: Model stores the training data points for distance calculation.
knn.fit(X_train_resampled, y_train_resampled)

# ==========================================
# Step 6: Evaluation
# ==========================================

# Why: Predict labels for the TEST set (which is Imbalanced and PCA-reduced).
# Output: Array of predictions (0s or 1s).
y_pred = knn.predict(X_test_pca)

print("\n--- Model Evaluation ---")
# Why: Standard confusion matrix to see True Positives, False Negatives, etc.
# Output: 2x2 Matrix.
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Why: Detailed report showing Precision, Recall, and F1-score for both classes.
# Crucial for imbalanced data (we care about Recall for interactions class '1').
# Output: Text table of metrics.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
