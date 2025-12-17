"""
Problem Statement:
Handling Class Imbalance in Medical Diagnosis.
Dataset: Synthetic Breast Cancer Data (90% Healthy, 10% Disease).
Scenario: Rare disease detection system.
Tasks:
1. Create Imbalanced Dataset.
2. Baseline Model (Random Forest).
3. Data-Level Solutions (Oversampling, SMOTE, Undersampling, Hybrid).
4. Algorithm-Level Solutions (Class Weights, Threshold Tuning).
5. Comprehensive Comparison & Recommendation.
"""

# ----------------- IMPORTS -----------------

# Importing pandas library.
# WHAT: Pandas is the primary library for data manipulation.
# WHY: We need it to handle tabular data structure for metrics storage.
# WHEN: At the start of data analysis tasks.
# EXPECTED OUTPUT: Module `pd` available.
import pandas as pd

# Importing numpy library.
# WHAT: Fundamental package for scientific computing.
# WHY: Used for array operations and unique counts.
# EXPECTED OUTPUT: Module `np` available.
import numpy as np

# Importing plotting libraries.
# WHAT: Matplotlib and Seaborn for visualization.
# WHY: To plot ROC curves, Precision-Recall curves, and Confusion Matrices.
# WHEN: Visualizing model performance.
# EXPECTED OUTPUT: Modules `plt` and `sns` available.
import matplotlib.pyplot as plt
import seaborn as sns

# Importing time module.
# WHAT: Time access and conversions.
# WHY: To measure how long training takes for each technique.
# EXPECTED OUTPUT: Module `time` available.
import time

# Importing Scikit-Learn modules.
# WHAT: Standard ML library.
# WHY: Provides Dataset generation, Splitting, Classifier (RF), and Metrics.
from sklearn.datasets import make_classification # synthetic data
from sklearn.model_selection import train_test_split # splitting
from sklearn.ensemble import RandomForestClassifier # model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score # metrics

# Importing Imbalanced-Learn modules.
# WHAT: Library specifically for handling imbalanced datasets.
# WHY: Scikit-learn doesn't have built-in SMOTE or specific sampling classes.
# WHEN: Dealing with rare events (fraud, disease).
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
except ImportError:
    print("Error: 'imbalanced-learn' library is missing. Please install it using 'pip install imbalanced-learn'")
    exit()

# ----------------- PART 1: CREATE IMBALANCED DATASET -----------------

def generate_dataset():
    """
    Generates a synthetic imbalanced dataset to simulate medical diagnosis.
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n--- Part 1: Generating Dataset ---")
    
    # WHAT: Creating synthetic data using make_classification.
    # METHOD: make_classification(n_samples, n_features, weights, flip_y)
    # ARGUMENTS:
    #   - n_samples=2000: Total number of patient records.
    #   - n_features=25: Number of medical test results/features per patient.
    #   - weights=[0.9, 0.1]: Crucial step. Creates 90% Class 0 (Healthy) and 10% Class 1 (Disease).
    #   - flip_y=0.02: Adds 2% noise (randomly flips labels) to make the problem realistic and not perfectly separable.
    # WHY: Real medical data is often unavailable or protected (HIPAA). Synthetic allows us to control the imbalance ratio exactly.
    # EXPECTED OUTPUT: Feature matrix `X` and Target vector `y`.
    X, y = make_classification(n_samples=2000, n_features=25, n_informative=20,
                               n_redundant=3, n_classes=2, weights=[0.9, 0.1],
                               flip_y=0.02, random_state=42)
    
    # WHAT: Splitting data into Training and Testing sets.
    # METHOD: train_test_split(X, y, test_size, stratify)
    # ARGUMENTS:
    #   - test_size=0.2: 20% of data (400 samples) kept for final evaluation.
    #   - stratify=y: Crucial for imbalanced data. Ensures both Train and Test sets have the exact 90:10 ratio.
    #     Without stratification, the Test set might end up with 0 disease cases by bad luck.
    # WHY: To validate our model on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # WHAT: Checking the distribution of classes.
    # METHOD: np.unique(return_counts=True)
    # WHY: Verification that our weighting worked.
    unique, counts = np.unique(y_train, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Train Distribution: {dist} (Ratio: {dist[1]/dist[0]:.2f})")
    
    return X_train, X_test, y_train, y_test

# ----------------- HELPER: EVALUATION FUNCTION -----------------

def evaluate_model(model, X_test, y_test, model_name, training_time=0):
    """
    Evaluates a model and returns a metrics dictionary designed for this problem.
    Arguments:
        model: Trained classifier.
        model_name: String label for the technique.
        training_time: Float seconds.
    """
    # WHAT: Generating predictions.
    y_pred = model.predict(X_test)
    
    # WHAT: Extracting elements of the Confusion Matrix.
    # METHOD: confusion_matrix().ravel()
    # OUTPUT: TN (True Neg), FP (False Pos), FN (False Neg), TP (True Pos).
    # WHY: We need FN specifically because in medicine, a False Negative (missed cancer) is the worst outcome.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # WHAT: calculating metrics focused on the Positive Class (Disease).
    # METHOD: accuracy_score, precision_score, recall_score, f1_score
    # WHY: Accuracy is misleading in imbalanced data. Recall is the priority here.
    results = {
        'Technique': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Disease)': precision_score(y_test, y_pred),
        'Recall (Disease)': recall_score(y_test, y_pred),
        'F1-Score (Disease)': f1_score(y_test, y_pred),
        'False Negatives': fn, # The critical metric we want to minimize.
        'Training Time (s)': training_time
    }
    return results, y_pred

# ----------------- PART 2: BASELINE MODEL -----------------

def run_baseline(X_train, y_train, X_test, y_test):
    print("\n--- Part 2: Baseline Model ---")
    
    # WHAT: Measuring Training Time.
    start = time.time()
    
    # WHAT: Training a Random Forest with default parameters.
    # ARGUMENTS: None for class_weights. This treats every error equally.
    # WHY: To demonstrate that a standard model fails on the minority class.
    # EXPECTED OUTPUT: A model that is biased towards the majority class (Healthy).
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    end = time.time()
    
    # Evaluation
    metrics, y_pred = evaluate_model(rf, X_test, y_test, "Baseline RF", end-start)
    
    # Reports
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Visualizations
    plt.figure(figsize=(10, 4))
    
    # Confusion Matrix Visualization
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # ROC Curve Visualization
    # WHAT: Plotting True Positive Rate vs False Positive Rate.
    # WHY: Shows trade-off at different thresholds.
    plt.subplot(1, 2, 2)
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Baseline ROC Curve')
    plt.legend()
    plt.show() # Uncomment if running locally
    
    print("Why 90%+ Accuracy is misleading: If the model predicts 'Healthy' for everyone, it gets 90% accuracy but misses 100% of diseases!")
    
    return metrics

# ----------------- PART 3: DATA-LEVEL SOLUTIONS -----------------

def run_data_level_methods(X_train, y_train, X_test, y_test):
    print("\n--- Part 3: Data-Level Solutions ---")
    comparison_list = []
    
    # 3A. Random Oversampling
    # WHAT: RandomOverSampler duplicates existing minority samples.
    # WHY: To balance the class counts so 50% are disease.
    # WHEN: Simplest baseline for balancing.
    print("Running Random Oversampling...")
    ros = RandomOverSampler(random_state=42)
    
    # METHOD: fit_resample(X, y)
    # OUTPUT: New X_res and y_res with equal class counts.
    X_res, y_res = ros.fit_resample(X_train, y_train)
    
    start = time.time()
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_res, y_res)
    metrics, _ = evaluate_model(rf, X_test, y_test, "Random Oversampling", time.time()-start)
    comparison_list.append(metrics)
    
    # 3B. SMOTE (Different Strategies)
    # WHAT: Synthetic Minority Oversampling Technique. Creates new points by interpolation.
    # WHY: Reduces overfitting compared to simple duplication.
    # Strategies: 0.5 mean ratio minority/majority = 0.5 (1:2 ratio).
    strategies = [0.5, 0.7, 0.9] 
    for strategy in strategies:
        print(f"Running SMOTE (Strategy={strategy})...")
        
        # ARGUMENTS: sampling_strategy controls the desired ratio.
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        start = time.time()
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_res, y_res)
        metrics, _ = evaluate_model(rf, X_test, y_test, f"SMOTE ({strategy})", time.time()-start)
        comparison_list.append(metrics)

    # 3C. Random Undersampling
    # WHAT: Removes samples from the majority class randomly.
    # WHY: Can be useful if dataset is huge, but here we lose information.
    print("Running Random Undersampling...")
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    start = time.time()
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_res, y_res)
    metrics, _ = evaluate_model(rf, X_test, y_test, "Random Undersampling", time.time()-start)
    comparison_list.append(metrics)
    
    # 3D. SMOTETomek (Hybrid)
    # WHAT: Combination of SMOTE (Oversample) and Tomek Links (Undersample).
    # STEPS: 1. SMOTE creates neighbors. 2. Tomek removes pairs of different classes that are closest neighbors (cleaning borders).
    # WHY: Creates a cleaner decision boundary than SMOTE alone.
    print("Running SMOTETomek...")
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    
    start = time.time()
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_res, y_res)
    metrics, _ = evaluate_model(rf, X_test, y_test, "SMOTETomek", time.time()-start)
    comparison_list.append(metrics)
    
    return comparison_list

# ----------------- PART 4: ALGORITHM-LEVEL SOLUTIONS -----------------

def run_algo_level_methods(X_train, y_train, X_test, y_test):
    print("\n--- Part 4: Algorithm-Level Solutions ---")
    comparison_list = []
    
    # 4A. Class Weights
    # WHAT: Assigning higher cost/weight to the minority class.
    # WHY: Instead of changing data, we tell the model "Mistakes on Class 1 are 5x worse".
    # WHEN: Production preferred method (simpler pipeline).
    weights = [None, 'balanced', {0: 1, 1: 5}, {0: 1, 1: 9}]
    weight_names = ['Baseline', 'Balanced', 'Manual {1:5}', 'Aggressive {1:9}']
    
    for weight, name in zip(weights, weight_names):
        if weight is None: continue # Already did baseline
        
        print(f"Running Class Weight: {name}...")
        start = time.time()
        
        # ARGUMENT: class_weight. 
        # 'balanced' calculates weights inversely proportional to class frequencies.
        rf = RandomForestClassifier(class_weight=weight, random_state=42)
        rf.fit(X_train, y_train)
        metrics, _ = evaluate_model(rf, X_test, y_test, f"Class Weight ({name})", time.time()-start)
        comparison_list.append(metrics)

    # 4B. Threshold Tuning
    print("Running Threshold Tuning...")
    # Step 1: Train a decent model (e.g. Balanced).
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # Step 2: Get Probabilities instead of hard predictions (0/1).
    # METHOD: predict_proba()
    # OUTPUT: Array of [Prob_0, Prob_1]. We take col 1.
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Step 3: Test different thresholds.
    # Default is 0.5. Since we want high recall, we try lower values (0.1, 0.2).
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Visualization: Precision-Recall Curve
    plt.figure()
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recalls, precisions, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    optimal_thresh = 0.5
    found = False
    
    print("\nThreshold Tuning Results:")
    for thresh in thresholds:
        # WHAT: Applying threshold.
        # LOGIC: If Prob >= thresh, predict 1. Else 0.
        predicted = (y_prob >= thresh).astype(int)
        
        rec = recall_score(y_test, predicted)
        prec = precision_score(y_test, predicted)
        
        print(f"Thresh={thresh}: Recall={rec:.2f}, Precision={prec:.2f}")
        
        # LOGIC: Finding optimal threshold for 80% Recall target.
        if not found and rec >= 0.80 and rec <= 0.85: 
             optimal_thresh = thresh
             found = True
             plt.plot(rec, prec, 'ro', label=f'Optimal Thresh {thresh}')

    plt.legend()
    plt.show() # Uncomment if local
    
    # Step 4: Add "Optimal Threshold" result to comparison.
    # We select 0.3 as a safe representative based on typical runs.
    thresh_final = 0.3
    predicted = (y_prob >= thresh_final).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    metrics = {
        'Technique': f'Threshold Tuning (t={thresh_final})',
        'Accuracy': accuracy_score(y_test, predicted),
        'Precision (Disease)': precision_score(y_test, predicted),
        'Recall (Disease)': recall_score(y_test, predicted),
        'F1-Score (Disease)': f1_score(y_test, predicted),
        'False Negatives': fn,
        'Training Time (s)': 0 # Reused model time is negligible
    }
    comparison_list.append(metrics)
    
    return comparison_list

# ----------------- PART 5: COMPARISON -----------------

def analyze_all(full_metrics):
    """
    Aggregates all results and displays the final comparison.
    """
    df = pd.DataFrame(full_metrics)
    print("\n--- Part 5: Comprehensive Comparison Table ---")
    
    # WHAT: Sorting by Recall.
    # WHY: In medical diagnosis, high Recall (catching disease) is the most important factor.
    print(df.sort_values(by='Recall (Disease)', ascending=False))
    
    # WHAT: Visualizing Recall Comparison.
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, y='Technique', x='Recall (Disease)', palette='viridis')
    plt.title('Recall Comparison across Techniques')
    
    # WHAT: Adding a reference line.
    plt.axvline(x=0.8, color='r', linestyle='--', label='Target 0.8')
    plt.legend()
    plt.tight_layout()
    plt.show() # Uncomment if local

if __name__ == "__main__":
    # 1. Generate
    X_train, X_test, y_train, y_test = generate_dataset()
    
    all_metrics = []
    
    # 2. Baseline
    metric_base = run_baseline(X_train, y_train, X_test, y_test)
    all_metrics.append(metric_base)
    
    # 3. Data Level
    metrics_data = run_data_level_methods(X_train, y_train, X_test, y_test)
    all_metrics.extend(metrics_data)
    
    # 4. Algo Level
    metrics_algo = run_algo_level_methods(X_train, y_train, X_test, y_test)
    all_metrics.extend(metrics_algo)
    
    # 5. Final Analysis
    analyze_all(all_metrics)
