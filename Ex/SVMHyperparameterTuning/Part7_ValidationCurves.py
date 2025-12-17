"""
Part7_ValidationCurves.py
Task: Validation Curves for C and Gamma.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from utils import load_and_preprocess_data

def run_validation_curves():
    print("\n=== Part 7: Validation Curves ===")
    
    data = load_and_preprocess_data()
    if data is None: return
    X_train, _, _, y_train, _, _ = data
    
    # Subsample for speed (curve calc is expensive)
    limit = int(len(X_train) * 0.2)
    X_sub = X_train[:limit]
    y_sub = y_train[:limit]
    
    # 1. C Validation Curve
    # WHAT: Measuring accuracy across range of C, while keeping gamma fixed (default).
    param_range = [0.01, 0.1, 1, 10, 100]
    print("Calculating C curve...")
    train_scores, test_scores = validation_curve(
        SVC(gamma='scale', random_state=42), 
        X_sub, y_sub, 
        param_name="C", 
        param_range=param_range,
        cv=3, 
        n_jobs=-1
    )
    
    # Averages
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(param_range, train_mean, label="Training score", color="darkorange", lw=2)
    plt.semilogx(param_range, test_mean, label="CV score", color="navy", lw=2)
    plt.title("Validation Curve (SVM, C Parameter)")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    
    # 2. Gamma Validation Curve
    param_range_g = [0.0001, 0.001, 0.01, 0.1, 1]
    print("Calculating Gamma curve...")
    train_scores_g, test_scores_g = validation_curve(
        SVC(C=10, random_state=42), 
        X_sub, y_sub, 
        param_name="gamma", 
        param_range=param_range_g,
        cv=3, 
        n_jobs=-1
    )
    
    train_mean_g = np.mean(train_scores_g, axis=1)
    test_mean_g = np.mean(test_scores_g, axis=1)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(param_range_g, train_mean_g, label="Training score", color="darkorange", lw=2)
    plt.semilogx(param_range_g, test_mean_g, label="CV score", color="navy", lw=2)
    plt.title("Validation Curve (SVM, Gamma Parameter)")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation_curves()
