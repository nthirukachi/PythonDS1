"""
Part3_ModelComparison.py
-------------------------------------------------------------------------------
Part 3: Implement and Compare 4 Algorithms

PROBLEM STATEMENT:
We need to benchmark different ML algorithms to find the best candidate for Fraud Detection.
Candidates are: KNN, SVM, Decision Tree, Random Forest.

STEPS TO SOLVE:
1. Load prepared data.
2. Train KNN (Nearest Neighbors).
3. Train SVM (Support Vector Machine) using a Subsample (Optimization).
4. Train Decision Tree.
5. Train Random Forest (Ensemble).
6. Compare metrics (Recall, Latency).

CONCEPTS & ARGUMENTS:
1. KNN (n_neighbors=5): Looks at 5 nearest points to decide class. Slow prediction.
2. SVM (class_weight='balanced'): Penalizes mistakes on Fraud class heavily.
3. Decision Tree (max_depth=10): Limits tree height to prevent overfitting.
4. Random Forest (n_estimators=100): Builds 100 trees and votes. Robust.

EXPECTED OUTPUT:
- Comparison Table showing Accuracy, Recall, and Prediction Time for each.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Part2_DataPrep import prepare_data

def evaluate_model(model, X_test, y_test, model_name, training_time):
    """
    Evaluates a trained model and returns metrics.
    
    ARGUMENTS:
    - model: The trained scikit-learn model object.
    - X_test: Data features to predict on.
    - y_test: Ground truth labels.
    - training_time: Time taken to train (measured outside).
    
    RETURNS:
    - Dictionary of metrics.
    """
    print(f"Evaluating {model_name}...")
    
    # 1. Measure Prediction Time
    start_pred = time.time()
    
    # WHAT: Generate predictions.
    y_pred = model.predict(X_test)
    
    pred_time = time.time() - start_pred
    
    # 2. Calculate Latency (Milliseconds per record)
    # FORMULA: (TotalTime / N_Samples) * 1000
    # WHY: In production, we need to know if it takes 1ms or 100ms per swipe.
    latency = (pred_time / len(X_test)) * 1000
    
    # 3. Calculate Metrics
    # Precision: TP / (TP + FP). Quality of alerts. (High = Few false alarms).
    # Recall: TP / (TP + FN). Quantity of fraud caught. (High = Few missed thefts).
    # F1-Score: Harmonic mean of Precision and Recall.
    # ROC-AUC: Area Under Curve. General measure of separability.
    res = {
        'Algorithm': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred),
        'Training Time (s)': training_time,
        'Pred Time/1k (s)': latency
    }
    # Log the most important metric
    print(f"-> Recall: {res['Recall']:.4f}")
    return res

def run_comparison():
    print("\n=== Part 3: Model Comparison ===")
    
    # Load prepared data
    data = prepare_data()
    if data is None: return
    X_train, X_val, X_test, y_train, y_val, y_test, _ = data
    
    results = []
    
    # ---------------------------------------------------------
    # 3A. k-Nearest Neighbors (KNN)
    # ---------------------------------------------------------
    print("\n[Training] KNN (n_neighbors=5)...")
    # WHAT: KNN Algorithm.
    # WHY: Simple baseline.
    # ARG: n_neighbors=5. Standard default.
    knn = KNeighborsClassifier(n_neighbors=5)
    
    start = time.time()
    # WHAT: Fit the model.
    # NOTE: KNN 'fit' is instantaneous (O(1)) as it just stores data.
    knn.fit(X_train, y_train) 
    results.append(evaluate_model(knn, X_test, y_test, 'KNN', time.time()-start))

    # ---------------------------------------------------------
    # 3B. Support Vector Machine (SVM)
    # ---------------------------------------------------------
    print("\n[Training] SVM (Subsampled 10%)...")
    # PROBLEM: SVM with RBF kernel has O(N^2) complexity.
    # 200,000^2 is 40 billion operations. It would take days to train.
    # SOLUTION: Train on a 10% Subsample (20k rows).
    limit = 20000
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # ARGUMENTS:
    # - class_weight='balanced': Crucial. Tells SVM to penalize mistakes on Fraud class 
    #   inversely proportional to frequency (e.g. 500x penalty).
    # - probability=True: Needed to calculate probability scores later.
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    
    start = time.time()
    svm.fit(X_sub, y_sub)
    results.append(evaluate_model(svm, X_test, y_test, 'SVM (Subsampled)', time.time()-start))

    # ---------------------------------------------------------
    # 3C. Decision Tree
    # ---------------------------------------------------------
    print("\n[Training] Decision Tree (max_depth=10)...")
    # ARGUMENTS:
    # - max_depth=10: Regularization. Prevents the tree from growing infinitely deep 
    #   and memorizing noise (overfitting).
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    start = time.time()
    dt.fit(X_train, y_train)
    results.append(evaluate_model(dt, X_test, y_test, 'Decision Tree', time.time()-start))

    # ---------------------------------------------------------
    # 3D. Random Forest
    # ---------------------------------------------------------
    print("\n[Training] Random Forest (100 trees)...")
    # WHAT: Ensemble of 100 Decision Trees.
    # ARGUMENTS:
    # - n_estimators=100: Number of trees.
    # - n_jobs=-1: Parallel processing. Uses all CPU cores to speed up training.
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    start = time.time()
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, 'Random Forest', time.time()-start))

    # ---------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results)
    print("\n--- Model Comparison Table ---")
    print(df_res)
    
    return df_res

if __name__ == "__main__":
    run_comparison()
