"""
====================================================================================================
1. PROBLEM STATEMENT:
Leakage-safe hyperparameter tuning with Pipeline + nested evaluation.

You are given a tabular dataset X with mixed feature scales and a binary label vector y with severe class imbalance (approx 1-5% positives).
We need to build a Support Vector Classifier (SVC) model that:
1.  Prevents "Data Leakage" (information from test/validation data sneaking into training).
2.  Handles mixed scales (requires Standardization).
3.  Optimizes hyperparameters (C, kernel, gamma) using Cross-Validation.
4.  Optimizes the Decision Threshold specifically for the F2-Score (which prioritizes Recall).

STEPS TO SOLVE THE PROBLEM:
1.  Data Generation: Create a synthetic dataset behaving like the problem description.
2.  Splitting: Set aside a "Held-Out" Test set that the model performs absolutely NO training or tuning on.
3.  Pipeline Construction: accurate combination of StandardScaler and SVC.
    -   Crucial: The scaler must fit ONLY on the training folds during cross-validation, not the whole dataset.
4.  Grid Search: Define the search space. Use StratifiedKFold to ensure every fold has enough positive samples.
5.  Training: Run the Grid Search.
6.  Threshold Tuning: Scan thresholds (0.1 to 0.9) to find the one giving the best F2 score.
7.  Final Evaluation: Report ROC-AUC, PR-AUC, and Confusion Matrix on the Test set.

EXPECTED OUTPUT:
-   Best parameters found by GridSearch.
-   Best Cross-Validation score.
-   A confusion matrix showing better Recall than a standard model.
-   Metrics (AUC ~0.9+, F2 ~High).
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================
# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the numpy library.
# 2.2: Why it is used: For high-performance array manipulations and math functions matching C-speed.
# 2.3: When to used: Start of script.
# 2.4: Where to use: Global scope.
# 2.5: How to use: standard alias `np`.
# 2.6: Output: None (Library loaded into memory).
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix, classification_report

# ==================================================================================================
# 1. DATA GENERATION
# ==================================================================================================
print("\n--- 1. Generating Data ---")

# Generate Synthetic Data
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates a random n-class classification problem.
# 2.2: Why it is used: To create a reproducible dataset when real data is unavailable.
# 2.3: When to used: Prototyping or testing logic.
# 2.4: Where to use: Data setup phase.
# 2.5: How to use: Assign return values to X (features) and y (target).
# 2.6: Output: X is (2000, 20) float array, y is (2000,) int array.

# 3. Arguments Explanation:
#    A. n_samples
#       3.1 What: Number of samples.
#       3.2 Why: Controls dataset size.
#       3.3 When to use: Always.
#       3.4 Where to use: Param 1.
#       3.5 How to use: Integer.
#       3.6 Sample Example: n_samples=1000 provides 1k rows.
#    B. weights
#       3.1 What: Class proportions.
#       3.2 Why: Creates imbalance (95% vs 5%).
#       3.3 When to use: For imbalance simulation.
#       3.4 Where to use: `weights` param.
#       3.5 How to use: List sum to 1.
#       3.6 Sample Example: [0.9, 0.1]
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, 
                           n_redundant=5, n_classes=2, weights=[0.95, 0.05], 
                           random_state=42)

# Simulate Mixed Scales
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Scales the 1st column by 1000x.
# 2.2: Why it is used: To simulate the "Mixed Scales" requirement of the problem statement.
# 2.3: When to used: During data creation.
# 2.4: Where to use: Before splitting.
# 2.5: How to use: Array broadcasting.
# 2.6: Output: The first column values now range roughly -3000 to +3000.
X[:, 0] = X[:, 0] * 1000 

print(f"Dataset Shape: {X.shape}")
print(f"Class Balance: {np.bincount(y)}")

# ==================================================================================================
# 2. SPLITTING (HOLD-OUT TEST SET)
# ==================================================================================================
print("\n--- 2. Splitting Data ---")

# train_test_split
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Splits arrays into random train and test subsets.
# 2.2: Why it is used: To prevent overfitting by holding out data the model never sees.
# 2.3: When to used: Before training.
# 2.4: Where to use: Preprocessing.
# 2.5: How to use: Unpack result into 4 variables.
# 2.6: Output: 4 Arrays: X_train (1600,20), X_test (400,20), y_train (1600,), y_test (400,).

# 3. Arguments Explanation:
#    A. X, y
#       3.1 What: Input data.
#       3.2 Why: Data to split.
#       3.3 When to use: Always.
#       3.4 Where to use: Args 1, 2.
#       3.5 How to use: Numpy arrays.
#       3.6 Sample Example: X, y
#    B. test_size
#       3.1 What: Size of test split.
#       3.2 Why: 0.2 gives good trade-off (80% train / 20% test).
#       3.3 When to use: Always.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: Float.
#       3.6 Sample Example: 0.2
#    C. stratify
#       3.1 What: Stratification target.
#       3.2 Why: Ensures 5% targets in both splits.
#       3.3 When to use: Classification.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: Pass y.
#       3.6 Sample Example: stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==================================================================================================
# 3. PIPELINE SETUP (LEAKAGE SAFETY)
# ==================================================================================================
print("\n--- 3. Setting up Pipeline ---")

# Pipeline
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates a pipeline object sequentially applying transforms and a final estimator.
# 2.2: Why it is used: To prevent data leakage by ensuring scaling happens *inside* CV folds.
# 2.3: When to used: When combining preprocessing and modeling.
# 2.4: Where to use: Model definition.
# 2.5: How to use: List of named steps.
# 2.6: Output: A Pipeline estimator object ready to fit.

# 3. Arguments Explanation:
#    A. steps
#       3.1 What: List of (name, transform) tuples.
#       3.2 Why: Defines the order of operations.
#       3.3 When to use: Instantiation.
#       3.4 Where to use: First arg.
#       3.5 How to use: [('name', Class())].
#       3.6 Sample Example: See code below.
#       - Sub-component: StandardScaler
#         3.1 What: Standardizes features by removing mean and scaling to unit variance.
#         3.2 Why: SVC is sensitive to scale; large variance features dominate the objective function.
#         3.3 When to use: Linear models, SVMs, Neural Nets.
#         3.4 Where to use: Step 1.
#         3.5 How to use: StandardScaler().
#         3.6 Sample Example: StandardScaler()
#       - Sub-component: SVC
#         3.1 What: C-Support Vector Classification.
#         3.2 Why: Good for high-dim tabular data.
#         3.3 When to use: Classification tasks.
#         3.4 Where to use: Step 2.
#         3.5 How to use: SVC().
#         - class_weight='balanced'
#           3.1 What: Adjusts weights inversely proportional to class frequencies.
#           3.2 Why: Forces model to penalty minority class errors heavily (solving imbalance).
#           3.3 When to use: Imbalanced data.
#           3.4 Where to use: Init arg.
#           3.5 How to use: String 'balanced'.
#           3.6 Sample Example: class_weight='balanced'
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, class_weight='balanced', random_state=42))
])

# ==================================================================================================
# 4. GRID SEARCH CONFIGURATION
# ==================================================================================================
print("\n--- 4. Configuring Grid Search ---")

# Param Grid
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines dictionary of parameters to try.
# 2.2: Why it is used: To explore the hyperparameter space.
# 2.3: When to used: With GridSearchCV.
# 2.4: Where to use: Variable def.
# 2.5: How to use: {'step__param': [values]}.
# 2.6: Output: Python Dictionary.
param_grid = {
    'svc__C': [0.1, 1, 10, 100],        
    'svc__kernel': ['linear', 'rbf'],   
    'svc__gamma': ['scale', 'auto']     
}

# StratifiedKFold
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Provides train/test indices to split data in train/test sets.
# 2.2: Why it is used: Stratified folds preserve class percentage (critical for imbalance).
# 2.3: When to used: Classification CV.
# 2.4: Where to use: Before GridSearch.
# 2.5: How to use: StratifiedKFold(n_splits=...).
# 2.6: Output: Cross-validation generator.

# 3. Arguments Explanation:
#    A. n_splits
#       3.1 What: Number of folds.
#       3.2 Why: 5 is robust standard.
#       3.3 When to use: Always.
#       3.4 Where to use: Arg.
#       3.5 How to use: Int.
#       3.6 Sample Example: 5
#    B. shuffle
#       3.1 What: Whether to shuffle before splitting.
#       3.2 Why: Randomizes order, removing data collection bias.
#       3.3 When to use: Almost always.
#       3.4 Where to use: Arg.
#       3.5 How to use: Bool True.
#       3.6 Sample Example: True
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Exhaustive search over specified parameter values for an estimator.
# 2.2: Why it is used: To determine Best Parameters (C, kernel, etc).
# 2.3: When to used: Hyperparameter tuning.
# 2.4: Where to use: Before training.
# 2.5: How to use: GridSearchCV(estimator, param_grid, ...).
# 2.6: Output: Unfitted GridSearchCV object.

# 3. Arguments Explanation:
#    A. estimator
#       3.1 What: Object to use.
#       3.2 Why: We optimize the pipeline.
#       3.3 When to use: Always.
#       3.4 Where to use: Arg 1.
#       3.5 How to use: Pass `pipe`.
#       3.6 Sample Example: pipe
#    B. scoring
#       3.1 What: Strategy to evaluate the performance of the cross-validated model.
#       3.2 Why: ROC-AUC is better than accuracy for imbalance.
#       3.3 When to use: Non-regression or Imbalanced tasks.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: String 'roc_auc'.
#       3.6 Sample Example: 'roc_auc'
#    C. n_jobs
#       3.1 What: Number of jobs to run in parallel.
#       3.2 Why: -1 uses all processors to speed up search.
#       3.3 When to use: Heavy computation.
#       3.4 Where to use: Keyword arg.
#       3.5 How to use: Int -1.
#       3.6 Sample Example: -1
grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=1, n_jobs=-1)

# ==================================================================================================
# 5. TRAINING
# ==================================================================================================
print("\n--- 5. Training Grid Search ---")

# grid.fit
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Runs fit on all parameter combinations and folds.
# 2.2: Why it is used: To execute the search.
# 2.3: When to used: After configuration.
# 2.4: Where to use: Execution phase.
# 2.5: How to use: grid.fit(X, y).
# 2.6: Output: Fits the grid object; stores `best_params_` and `best_score_`.
grid.fit(X_train, y_train)

print(f"\nBest Params: {grid.best_params_}")
print(f"Best CV Score (ROC-AUC): {grid.best_score_:.4f}")

# ==================================================================================================
# 6. THRESHOLD OPTIMIZATION (F-beta = 2)
# ==================================================================================================
print("\n--- 6. Optimizing Threshold (F-beta=2) ---")

best_model = grid.best_estimator_

# predict_proba
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Predict class probabilities for X_test.
# 2.2: Why it is used: To get granular confidence scores for threshold tuning.
# 2.3: When to used: Evaluation.
# 2.4: Where to use: Before threshold loop.
# 2.5: How to use: predict_proba(X) and slice [:, 1] for positive class.
# 2.6: Output: Array of float probabilities (0.0 to 1.0) for positive class.
y_probs = best_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f2 = 0.0

for thresh in thresholds:
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Thresholds probability to binary 0/1.
    # 2.6: Output: Binary array (0 or 1).
    y_pred_temp = (y_probs >= thresh).astype(int)
    
    # fbeta_score
    # ----------------------------------------------------------------------------------------------
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Compute the F-beta score.
    # 2.2: Why it is used: To evaluate effectiveness with Recall weighted higher.
    # 2.3: When to used: Imbalanced evaluation.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: fbeta_score(true, pred, beta=2).
    # 2.6: Output: Float score.

    # 3. Arguments:
    #    A. beta
    #       3.1 What: Weight of recall vs precision.
    #       3.2 Why: 2.0 biases metric towards Recall.
    #       3.3 When to use: Cost of FN > Cost of FP.
    #       3.4 Where to use: Keyword arg.
    #       3.5 How to use: Float > 0.
    #       3.6 Sample Example: 2
    score = fbeta_score(y_test, y_pred_temp, beta=2)
    
    if score > best_f2:
        best_f2 = score
        best_threshold = thresh

print(f"Optimal Threshold for F2: {best_threshold:.3f}")
print(f"Max F2 Score on Test: {best_f2:.4f}")

# ==================================================================================================
# 7. FINAL EVALUATION
# ==================================================================================================
print("\n--- 7. Final Test Evaluation ---")

y_final_pred = (y_probs >= best_threshold).astype(int)

# Metrics
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Calculates AUC, PR-AUC, F1.
# 2.6: Output: Float values representing model quality.
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
f1_final = fbeta_score(y_test, y_final_pred, beta=1)

print(f"Final Test ROC-AUC: {roc_auc:.4f}")
print(f"Final Test PR-AUC: {pr_auc:.4f}")
print(f"Final Test F1-Score: {f1_final:.4f}")

# Confusion Matrix
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Computes confusion matrix to evaluate the accuracy of a classification.
# 2.2: Why it is used: To visualize TP, FP, TN, FN.
# 2.3: When to used: Final reporting.
# 2.4: Where to use: End of script.
# 2.5: How to use: confusion_matrix(y_true, y_pred).
# 2.6: Output: ndarray of shape (2, 2).
cm = confusion_matrix(y_test, y_final_pred)
print(cm)
print(f"\nBreakdown: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_final_pred))
