"""
Part2_DataLevel.py
-------------------------------------------------------------------------------
Part 2: Data-Level Solutions (Resampling)

PROBLEM STATEMENT:
If the model bias comes from data imbalance, one solution is to fix the data.
We can:
1.  **Undersample**: Delete healthy patients (Class 0) until n(0) == n(1).
2.  **Oversample (SMOTE)**: Create fake sick patients (Class 1) until n(1) == n(0).

STEPS TO SOLVE:
1. Import `imblearn` library.
2. Apply `RandomUnderSampler` to Training Data.
3. Apply `SMOTE` (Synthetic Minority Over-sampling Technique) to Training Data.
4. Train models on these "Balanced" datasets.
5. Evaluate on the ORIGINAL Test set (Never resample test data!).

EXPECTED OUTPUT:
- Undersampling: High Recall, Low Precision (High false alarms).
- SMOTE: Better balance of Precision/Recall.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from utils import generate_imbalanced_data

# WHAT: Import resampling algorithms.
# These require `pip install imbalanced-learn`.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False
    print("WARNING: `imblearn` not found. Skipping Part 2 execution.")

def evaluate_resampling(X_train, y_train, X_test, y_test, method_name, sampler=None):
    """
    Helper function to run the pipeline: Resample -> Train -> Test.
    
    ARGS:
    - sampler: The imblearn object (e.g. SMOTE). If None, does nothing.
    """
    print(f"\n--- Testing Method: {method_name} ---")
    
    # 1. Resample Phase
    # WHAT: Modify the Training Data only.
    if sampler:
        print(f"Resampling data...")
        # METHOD: .fit_resample(X, y). Returns new arrays with equal class counts.
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"New Training Shape: {X_res.shape} (Balanced)")
    else:
        # No resampling (Control group).
        X_res, y_res = X_train, y_train
        
    # 2. Training Phase
    # WHAT: Train a fresh Random Forest on the balanced data.
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_res, y_res)
    
    # 3. Evaluation Phase
    # WHAT: Predict on the ORIGINAL X_test (Real World Distribution).
    y_pred = clf.predict(X_test)
    
    # Metrics
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Recall (Class 1): {rec:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    
    return {'Method': method_name, 'Recall': rec, 'F1': f1}

def run_part2_datalevel():
    print("\n=== Part 2: Data-Level Solutions (Resampling) ===")
    
    if not IMBLEARN_OK:
        return []

    X_train, X_test, y_train, y_test = generate_imbalanced_data()
    
    results = []
    
    # Strategy A: Random Undersampling
    # WHAT: Randomly discards majority samples.
    # PRO: Very fast. 
    # CON: Loses valuable information.
    rus = RandomUnderSampler(random_state=42)
    results.append(evaluate_resampling(X_train, y_train, X_test, y_test, 'Undersampling', rus))
    
    # Strategy B: SMOTE
    # WHAT: Generates synthetic points between minority neighbors.
    # PRO: Keeps all original data.
    # CON: Can create noise if classes overlap.
    smote = SMOTE(random_state=42)
    results.append(evaluate_resampling(X_train, y_train, X_test, y_test, 'SMOTE', smote))
    
    # Strategy C: SMOTE + Tomek
    # WHAT: Apply SMOTE, then use Tomek links to remove "confusing" boundary points.
    # PRO: Cleaner decision boundary.
    smt = SMOTETomek(random_state=42)
    results.append(evaluate_resampling(X_train, y_train, X_test, y_test, 'SMOTE + Tomek', smt))
    
    # Summarize
    print("\nResampling Comparison:")
    print(pd.DataFrame(results))
    
    return results

if __name__ == "__main__":
    run_part2_datalevel()
