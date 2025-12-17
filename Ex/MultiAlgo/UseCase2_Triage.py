"""
UseCase2_Triage.py
-------------------------------------------------------------------------------
USE CASE 2: Emergency Department (ED) Triage

PROBLEM STATEMENT:
In an Emergency Room, thousands of patients arrive. We must assign them a priority
level (1 to 5) instantly. 
This is a **Multi-Class Classification** problem.
Critical Constraint: **Latency**. The prediction must happen in real-time (<2 seconds) 
to not slow down the intake process.

STEPS TO SOLVE:
1. Load Synthetic 5-Class Data (from utils).
2. Train 3 candidate models:
   - Random Forest (Baseline Accuracy).
   - Linear SVM (Known for speed).
   - KNN (Simple, but lazy learner).
3. Measure **Inference Time** (Prediction Latency).
4. Evaluate using **Macro-F1 Score** (Average fairness across all 5 classes).
5. Plot Confusion Matrix for the best model.

EXPECTED OUTPUT:
- Comparison Table showing Latency (ms) and Accuracy.
- A textual confirmation if latency < 2s.
- A Confusion Matrix PNG.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# WHAT: Import Multi-Class Classifiers.
# LinearSVC: Very fast implementation of SVM using liblinear.
# KNeighbors: Remembers all training data points.
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# WHAT: Metric.
# F1 Score: Harmonic mean of Precision and Recall.
# confusion_matrix: Detailed breakdown of errors.
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from utils import generate_triage_data, plot_confusion_matrix

def run_triage_use_case():
    print("\n=== USE CASE 2: ED TRIAGE (MULTI-CLASS) ===")
    
    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    X, y = generate_triage_data() 
    # NOTE: y contains values [0, 1, 2, 3, 4] representing Triage Levels.
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # ---------------------------------------------------------
    # 2. Model Definition
    # ---------------------------------------------------------
    models = {
        # Model A: Random Forest
        # ARG: n_estimators=50. Reduced from 100 to speed up latency.
        'Random Forest': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42),
        
        # Model B: Linear SVM
        # ARG: dual='auto'. Handles optimization selection automatically.
        # WHY: Linear SVC is much faster than SVC(kernel='rbf') for multi-class.
        'Linear SVM': LinearSVC(class_weight='balanced', random_state=42, dual='auto'),
        
        # Model C: KNN
        # ARG: n_neighbors=5. 
        # WARNING: KNN is "lazy". Training is fast, but Prediction is slow (searches all points).
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    
    # ---------------------------------------------------------
    # 3. Training & Latency Testing
    # ---------------------------------------------------------
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Fit phase
        model.fit(X_train, y_train)
        
        # LATENCY TEST START
        start = time.time()
        
        # Prediction phase (Where speed matters)
        y_pred = model.predict(X_test)
        
        # LATENCY TEST END
        total_time = time.time() - start
        
        # WHAT: Calculate per-sample latency involved.
        # Math: (TotalSeconds / NumSamples) * 1000 = Milliseconds per Sample.
        latency_ms = (total_time / len(X_test)) * 1000
        
        # WHAT: Calculate Accuracy (F1).
        # ARG: average='macro'. 
        # WHY: Standard accuracy might be biased if Class 3 is huge. Macro averages all 5 scores.
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'Algorithm': name, 
            'Macro F1': f1, 
            'Latency (ms)': latency_ms,
            'Meets <2s Reqt?': "YES" if total_time < 2.0 else "NO"
        })
        
        # ---------------------------------------------------------
        # 4. Visualization (Confusion Matrix)
        # ---------------------------------------------------------
        # Generate plot for the Random Forest model as a sample
        if name == 'Random Forest':
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, labels=['L1', 'L2', 'L3', 'L4', 'L5'], title=f'{name} Triage Matrix')

    print("\nComparison Table:")
    print(pd.DataFrame(results))
    
    print("\nCONCLUSION FOR USE CASE 2:")
    print("- Linear SVM is usually fastest (lowest latency) due to simple dot-product math.")
    print("- KNN can be slow at inference time because it calculates distance to all points.")
    print("- All models meet the <2s requirement on this specific dataset size.")

if __name__ == "__main__":
    run_triage_use_case()
