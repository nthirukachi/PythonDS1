"""
Problem Statement:
SVM and Decision Tree Implementation with Class Imbalance (Breast Cancer Dataset).

Objectives:
Part A: Implementation.
1. Creates a 90:10 Imbalanced Dataset (90% Benign, 10% Malignant).
   (Note: We undersample Malignant class to achieve this ratio, correcting the prompt's instruction which would have achieved the inverse).
2. Implement 6 Models:
   - SVM (Baseline, Balanced, Custom Weights).
   - Decision Tree (Baseline, Balanced, Custom Weights).

Part B: Evaluation.
1. Metrics: Precision, Recall, F1, ROC-AUC.
2. Visualizations: Heatmaps, ROC Curves, PR Curves, Metric Bars.
3. Analysis: Recommendation and threshold discussion.
"""

# What: Import NumPy.
# Why: Essential for array operations, specifically handling indices for undersampling.
# When: Start of script.
# Output: Module 'numpy' as 'np'.
import numpy as np

# What: Import Pandas.
# Why: Used to create and display the final performance table structure.
# When: Start of script.
# Output: Module 'pandas' as 'pd'.
import pandas as pd

# What: Import Matplotlib.
# Why: Required for generating ROC curves and Bar Charts.
# When: Start of script.
# Output: Module 'pyplot' as 'plt'.
import matplotlib.pyplot as plt

# What: Import Seaborn.
# Why: Used specifically for the Heatmap visualization of Confusion Matrices.
# When: Start of script.
# Output: Module 'seaborn' as 'sns'.
import seaborn as sns

# What: Import Dataset Loader.
# Why: Provides the source Breast Cancer data.
# Output: Function 'load_breast_cancer'.
from sklearn.datasets import load_breast_cancer

# What: Import Classifiers.
# Why: SVC (Support Vector Classifier) and Decision Tree are the requirements.
# Output: Classes available.
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# What: Import Splitter.
# Why: To separate Train/Test sets while preserving imbalance (Stratify).
# Output: Function 'train_test_split'.
from sklearn.model_selection import train_test_split

# What: Import Metrics.
# Why: We need a comprehensive suite: CM, Precision, Recall, F1, ROC-AUC to evaluate imbalance performance.
# Output: Multiple metric functions.
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                             average_precision_score, roc_curve, precision_recall_curve, f1_score)

# ==========================================
# Part A: Dataset Creation (Imbalance 90:10)
# ==========================================

print("--- 1. Data Preparation (90:10 Imbalance) ---")

# Load Data
# What: Retrieve the raw dataset.
# Output: data.data is (569, 30), data.target is (569,).
data = load_breast_cancer()
X_raw = data.data
y_raw = data.target

# Identify indices
# What: Find row numbers where target is 0 (Malignant) and where target is 1 (Benign).
# Why: We need to manipulate these specific groups to change the ratio.
# Output: Two arrays of integer indices.
idx_0 = np.where(y_raw == 0)[0] # Malignant
idx_1 = np.where(y_raw == 1)[0] # Benign

# Calculate Target Counts
# Logic: We keep ALL Benign samples (357). 
# We want Benign to represent 90% of the final dataset.
# Formula: Total_Final = Benign_Count / 0.9.
# Malignant_Final = Total_Final - Benign_Count = (Benign/0.9) * 0.1.
n_benign = len(idx_1)
n_malignant_target = int(n_benign / 0.9 * 0.1)

print(f"Original Counts: Benign={len(idx_1)}, Malignant={len(idx_0)}")
print(f"Target Counts:   Benign={n_benign}, Malignant={n_malignant_target}")

# Undersample Malignant
# What: Set seed for reproducibility.
# Why: Ensures we pick the same random 39 patients every time.
# When: Before random choice.
np.random.seed(42)

# What: Randomly select 'n_malignant_target' indices from the 'idx_0' list.
# Why: This physically removes the majority of Malignant cases to create the artificial imbalance.
# Output: Array of 39 indices.
idx_0_undersampled = np.random.choice(idx_0, size=n_malignant_target, replace=False)

# Combine Indices
# What: Merge the 357 Benign indices and the 39 Undersampled Malignant indices.
# Output: Array of 396 total indices.
idx_final = np.concatenate([idx_1, idx_0_undersampled])

# Create Final Arrays
# What: Filter the original X and y arrays using the selected indices.
# Why: Creates the actual Imbalanced Dataset for training.
# Output: X (396, 30), y (396,).
X = X_raw[idx_final]
y = y_raw[idx_final]

# Split Train/Test
# What: 80/20 split, stratified to maintain the 90:10 ratio in chunks.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Final Dataset Shape: {X.shape}")
print(f"Final Class Distribution: {np.bincount(y)} (0: Malignant, 1: Benign)")

# ==========================================
# Part A: Model Configurations
# ==========================================

# Define Models dictionary
# What: Create a Dictionary mapping Names (strings) to Model Objects (instantiated classes).
# Why: Allows us to iterate cleanly through all 6 permutations in the training loop.
# Output: Dictionary of {'ModelName': classifier_object}.
models = {
    # 1. SVM Baseline
    # What: Standard Linear SVM.
    # Why: Control group to show poor performance on imbalance.
    'SVM_Baseline': SVC(kernel='linear', probability=True, random_state=42),

    # 2. SVM Balanced
    # What: SVM with 'class_weight="balanced"'.
    # Why: Automatically adjusts weights inversely proportional to class frequencies.
    'SVM_Balanced': SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),

    # 3. SVM Custom
    # What: SVM with manual weights {0: 9, 1: 1}.
    # Why: Since Malignant (0) is 10% and Benign (1) is 90%, we boost Malignant importance by 9x to equalize influence.
    'SVM_Custom':   SVC(kernel='linear', class_weight={0: 9, 1: 1}, probability=True, random_state=42),
    
    # 4. Decision Tree Baseline
    # What: Standard Tree.
    'DT_Baseline':  DecisionTreeClassifier(random_state=42),

    # 5. DT Balanced
    # What: Tree with auto-balancing weights.
    'DT_Balanced':  DecisionTreeClassifier(class_weight='balanced', random_state=42),

    # 6. DT Custom
    # What: Tree with manual 9:1 weights.
    'DT_Custom':    DecisionTreeClassifier(class_weight={0: 9, 1: 1}, random_state=42)
}

# Store Results
results = []

# Training Loop
print("\n--- 2. Training & Comparison ---")

# What: Create a large figure to hold 6 subplots (confusion matrices).
# Arguments: 18x10 inches.
plt.figure(figsize=(18, 10)) 

# Loop through models
# What: Enumerate gives us index 'i' (0-5) and key-value pair (name, model).
for i, (name, model) in enumerate(models.items()):
    # What: Train the specific model.
    model.fit(X_train, y_train)
    
    # Predict Label
    # What: Get class 0/1 predictions.
    # Output: Array of 0s and 1s.
    y_pred = model.predict(X_test)
    
    # Predict Probability
    # What: Get probability of Malignant Class (0).
    # Why: Needed for ROC-AUC score.
    # Logic: [:, 0] selects the probability column for class 0.
    y_prob = model.predict_proba(X_test)[:, 0] 
    
    # Calculate Metrics (Focus on Class 0 - Malignant)
    # What: Generate Confusion Matrix.
    # Arguments: labels=[0, 1] fixes the order so [0,0] is True Positive (Malignant).
    # Output: 2x2 Array [[TP, FN], [FP, TN]].
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # What: Calculate Recall (Sensitivity).
    # Formula: True Positives / (True Positives + False Negatives).
    # Why: Standard metric for 'Missing Disease'.
    recall = cm[0,0] / (cm[0,0] + cm[0,1])
    
    # What: Calculate Precision.
    # Formula: TP / (TP + FP).
    # Control against DivideByZero if model predicts no positives.
    precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    
    # What: Calculate F1 Score.
    # Formula: Harmonic Mean of Precision and Recall.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # What: Calculate ROC-AUC.
    # Arguments: y_test == 0 converts Ground Truth to Boolean (True if Malignant).
    roc_auc = roc_auc_score(y_test == 0, y_prob)

    # Store in List
    results.append({
        'Model': name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CM': cm,
        'Prob': y_prob
    })
    
    # Plot Confusion Matrix (Subplot)
    # What: Activate the i-th subplot (1 to 6).
    plt.subplot(2, 3, i+1)
    
    # What: Draw Heatmap.
    # Arguments: 
    # - annot=True: Write numbers in cells.
    # - fmt='d': Format as integers.
    # - cmap='Blues': Blue color scheme.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant(0)', 'Benign(1)'], 
                yticklabels=['Malignant(0)', 'Benign(1)'])
    
    # What: Add title with Recall Score.
    plt.title(f"{name}\nRecall: {recall:.2f}")

# What: Adjust spacing so subplots don't overlap.
plt.tight_layout()

# What: Show the 6-grid plot.
# When: End of training loop.
# plt.show()

# ==========================================
# Part B: Evaluation & Analysis
# ==========================================

# 1. Overlay ROC Curves
# What: Visualize Tradeoff between TPR and FPR.
# Arguments: 10x6 inches.
plt.figure(figsize=(10, 6))

# What: Iterate through stored results to plot lines.
for res in results:
    # What: Calculate ROC Curve Points.
    # Arguments: 
    # - y_test == 0: Convert labels to True (Malignant) vs False.
    # - res['Prob']: Probability scores for the Positive class.
    # Output: fpr (x-axis), tpr (y-axis).
    fpr, tpr, _ = roc_curve(y_test == 0, res['Prob'])
    
    # What: Plot the line for this model.
    # Label: Shows Model Name and AUC Score.
    plt.plot(fpr, tpr, label=f"{res['Model']} (AUC={res['ROC-AUC']:.2f})")

# What: Draw Diagonal Line (Random Guessing).
# Why: Baseline. Any model below this line is worse than random.
plt.plot([0, 1], [0, 1], 'k--')

# What: Add Labels.
plt.xlabel('False Positive Rate (Benign flagged as Malignant)')
plt.ylabel('True Positive Rate (Malignant Caught)')
plt.title('ROC Curves (Target: Class 0 Malignant)')
plt.legend()
plt.grid(True)
# plt.show()

# 2. Results DataFrame
# What: Convert dictionary list to Pandas Table.
# Why: Cleaner display.
# Output: DataFrame.
df_res = pd.DataFrame(results)

# What: Drop Complex columns (Matrix and Probability array) for display.
# Output: Simplified table.
df_res = df_res.drop(columns=['CM', 'Prob'])

print("\n--- Final Performance Table (Class 0 Focus) ---")
print(df_res.to_string(index=False))

# 3. Bar Chart Comparison
# What: Select specific columns for plotting.
# Why: To visually compare key metrics.
# Output: Bar plot object.
ax = df_res.set_index('Model')[['Recall', 'F1-Score', 'ROC-AUC']].plot(kind='bar', figsize=(12, 6))

# What: Customize Chart.
plt.title('Key Metrics Comparison')
plt.ylabel('Score')

# What: Add horizontal grid lines.
plt.grid(True, axis='y')

# What: Rotate x-axis labels for readability.
plt.xticks(rotation=45)

# What: Fix layout.
plt.tight_layout()

# What: Render plot.
# plt.show()

# ==========================================
# Analysis Text
# ==========================================

print("\n" + "="*50)
print("PART B: EVALUATION ANALYSIS")
print("="*50)

print("""
1. Best Model Recommendation:
   - SVM_Custom (Weight 9:1) or SVM_Balanced are typically superior.
   - Why: In the imbalanced scenario, the Baseline SVM often ignores the minority class (Recall ~0). 
     The Weighted/Balanced SVM forces the boundary to respect the few Malignant cases, boosting Recall to >0.8 or 1.0.

2. Metrics Justification:
   - Accuracy is misleading. A model predicting "All Benign" gets 90% accuracy but 0% Recall.
   - Recall is King. We CANNOT miss a cancer diagnosis.
   - F1-Score is Queen. We need to measure the balance so we don't flag everyone as sick (Precision).

3. Optimal Classification Threshold:
   - Default threshold is 0.5 probability.
   - Suggestion: Lower likelihood threshold to 0.3 or 0.2.
   - Why: Even if the model is only 30% sure it's malignant, it is safer to flag it for a human doctor review than to dismiss it. This increases Recall at the cost of Precision.
""")
