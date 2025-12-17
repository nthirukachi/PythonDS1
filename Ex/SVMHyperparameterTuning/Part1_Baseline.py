"""
Part1_Baseline.py
Task: Data Preparation and Baseline SVM Model.
"""

# ----------------- IMPORTS -----------------
import time
from sklearn.svm import SVC
from utils import load_and_preprocess_data, evaluate_model

# ----------------- MAIN EXECUTION -----------------
def run_baseline():
    print("\n=== Part 1: Baseline SVM ===")
    
    # WHAT: Load data using our shared utility.
    # WHY: Ensures consistent splits across all scripts.
    data = load_and_preprocess_data()
    if data is None: return
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # WHAT: Initialize Baseline SVM.
    # ARGUMENTS: C=1.0 (default regularization), gamma='scale' (default kernel coefficient).
    # WHY: Sets a benchmark performance.
    clf = SVC(C=1.0, gamma='scale', random_state=42)
    
    print("Training Baseline Model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # WHAT: Evaluation.
    metrics = evaluate_model(clf, X_test, y_test, "Test")
    
    print(f"\nBaseline Results (Training Time: {train_time:.4f}s):")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1']:.4f}")
    
    return metrics

if __name__ == "__main__":
    run_baseline()
