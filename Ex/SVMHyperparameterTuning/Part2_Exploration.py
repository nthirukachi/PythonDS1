"""
Part2_Exploration.py
Task: Understand Hyperparameters Through Experimentation (C and Gamma loops).
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from utils import load_and_preprocess_data

def run_exploration():
    print("\n=== Part 2: Hyperparameter Exploration ===")
    
    data = load_and_preprocess_data()
    if data is None: return
    X_train, _, _, y_train, _, _ = data # We use Validation set for this part?
    # Actually, prompt says "Train on train set, Evaluate on validation set".
    # So we need X_val, y_val.
    X_train, X_val, X_test, y_train, y_val, y_test = data

    # ----------------- 2A. C PARAMETER EXPLORATION -----------------
    print("\n--- 2A. Exploring C Parameter (Fixed gamma=0.01) ---")
    
    # WHAT: range of C values to test.
    # C controls the trade-off. Large C = Low bias/High variance (strict). Small C = High bias/Low variance (smooth).
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    
    results_c = []
    
    for C_val in c_values:
        print(f"Testing C={C_val}...")
        start = time.time()
        
        # WHAT: Train SVM with specific C.
        clf = SVC(C=C_val, gamma=0.01, random_state=42)
        clf.fit(X_train, y_train)
        
        train_time = time.time() - start
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        
        results_c.append({
            'C': C_val,
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Training Time': train_time
        })

    df_c = pd.DataFrame(results_c)
    print(df_c)
    
    # PLOT: C vs Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogx(df_c['C'], df_c['Train Accuracy'], marker='o', label='Train')
    plt.semilogx(df_c['C'], df_c['Validation Accuracy'], marker='s', label='Validation')
    plt.title('C vs Accuracy (gamma=0.01)')
    plt.xlabel('C parameter (log scale)')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # PLOT: C vs Time
    plt.subplot(1, 2, 2)
    plt.semilogx(df_c['C'], df_c['Training Time'], marker='x', color='r')
    plt.title('C vs Training Time')
    plt.xlabel('C parameter')
    plt.ylabel('Time (s)')
    plt.tight_layout()
    plt.show() # Remove if headless

    # ----------------- 2B. GAMMA PARAMETER EXPLORATION -----------------
    print("\n--- 2B. Exploring Gamma Parameter (Fixed C=10) ---")
    
    # WHAT: range of Gamma values.
    # Gamma defines how far the influence of a single training example reaches. 
    # Low = Far, High = Close (complex, jagged boundary).
    gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    
    results_g = []
    
    for g_val in gamma_values:
        print(f"Testing Gamma={g_val}...")
        
        clf = SVC(C=10, gamma=g_val, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        
        results_g.append({
            'Gamma': g_val,
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc
        })
        
    df_g = pd.DataFrame(results_g)
    print(df_g)
    
    # PLOT: Gamma vs Accuracy
    plt.figure(figsize=(6, 4))
    plt.semilogx(df_g['Gamma'], df_g['Train Accuracy'], marker='o', label='Train')
    plt.semilogx(df_g['Gamma'], df_g['Validation Accuracy'], marker='s', label='Validation')
    plt.title('Gamma vs Accuracy (C=10)')
    plt.xlabel('Gamma (log scale)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_exploration()
