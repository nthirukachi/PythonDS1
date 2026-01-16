# ----------------------------------------------------------------------------------
# PROBLEM STATEMENT:
# Design a monitoring plan for a production ML system with:
# (i)   2 Data-Quality Checks (Nulls, Range)
# (ii)  2 Drift Checks (Distribution-based: KS Test, Mean Shift)
# (iii) 1 Alert Rule (Plain language action)
# ----------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# ==================================================================================
# 1. SYNTHETIC DATA GENERATION: PRODUCING BATCHES
# ==================================================================================
# We will simulate 3 batches of data:
# Batch 0: Training/Baseline Data (Normal behavior)
# Batch 1: Data Drift (Input feature P(X) changes)
# Batch 2: Concept Drift (Relationship P(Y|X) changes)
# ==================================================================================

def generate_data(batch_type='baseline', n_samples=1000, random_state=42):
    """
    Generates synthetic dataset for a binary classification problem.
    Features: 'income' (cont), 'age' (opt, but we will use income mainly).
    Target: 'loan_approved' (0 or 1).
    """
    np.random.seed(random_state)
    
    # --- BASELINE BATCH (Training Data) ---
    if batch_type == 'baseline':
        # Feature X: Income (Normal distribution: Mean=50k, Std=10k)
        income = np.random.normal(50000, 10000, n_samples)
        # Relationship P(Y|X): Probability of approval increases with income
        # Simple Logic: If Income > 52000, high chance of approval
        # Add some noise
        prob = 1 / (1 + np.exp(-(income - 50000) / 10000))
        approval = (np.random.rand(n_samples) < prob).astype(int)
        
        return pd.DataFrame({'income': income, 'loan_approved': approval})

    # --- DAT DRIFT BATCH (P(X) changes) ---
    elif batch_type == 'data_drift':
        # Feature X Shift: The economy booms! Everyone earns more.
        # X Changes: Mean shifts from 50k -> 75k.
        income = np.random.normal(75000, 15000, n_samples)
        
        # Relationship P(Y|X) STAYS THE SAME (Model Logic is still valid)
        prob = 1 / (1 + np.exp(-(income - 50000) / 10000))
        approval = (np.random.rand(n_samples) < prob).astype(int)
        
        return pd.DataFrame({'income': income, 'loan_approved': approval})

    # --- CONCEPT DRIFT BATCH (P(Y|X) changes) ---
    elif batch_type == 'concept_drift':
        # Feature X is Normal (Back to 50k Mean)
        income = np.random.normal(50000, 10000, n_samples)
        
        # Relationship P(Y|X) CHANGES (Policy Change)
        # Now, stricter lending! You need MUCH higher income to get approved.
        # Threshold implies effectively mean needs to be around 65000 to get 50% chance
        prob = 1 / (1 + np.exp(-(income - 65000) / 10000)) 
        approval = (np.random.rand(n_samples) < prob).astype(int)
        
        return pd.DataFrame({'income': income, 'loan_approved': approval})
    
    else:
        raise ValueError("Unknown batch_type")

# ==================================================================================
# 2. MONITORING SYSTEM FUNCTIONS
# ==================================================================================

def check_data_quality(df):
    """
    (i) 2 Data Quality Checks:
    1. Null Value Check: Are there any missing values?
    2. Range Check: Is 'income' non-negative?
    """
    results = {}
    
    # Check 1: Null Check
    null_count = df.isnull().sum().sum()
    results['null_check_pass'] = (null_count == 0)
    results['null_count'] = null_count
    
    # Check 2: Range Check (Income must be >= 0)
    min_income = df['income'].min()
    results['range_check_pass'] = (min_income >= 0)
    results['min_value'] = min_income
    
    return results

def check_drift(baseline_df, current_df):
    """
    (ii) 2 Drift Checks (Distribution-based):
    1. Kolmogorov-Smirnov (KS) Test: Compares probability distributions of 'income'.
    2. Mean Shift Check: Compares statistical mean of 'income'.
    """
    results = {}
    
    # Check 1: KS Test
    # H0: Two samples are drawn from the same distribution.
    # If p_value < 0.05, we reject H0 -> Drift Detected.
    stat, p_value = ks_2samp(baseline_df['income'], current_df['income'])
    results['ks_stat'] = stat
    results['ks_p_value'] = p_value
    results['drift_detected_ks'] = (p_value < 0.05)
    
    # Check 2: Mean Shift (Simple statistic check)
    # If mean varies by more than 20% of baseline mean, flag it.
    mean_base = baseline_df['income'].mean()
    mean_curr = current_df['income'].mean()
    perc_diff = abs(mean_curr - mean_base) / mean_base
    results['mean_diff_perc'] = perc_diff
    results['drift_detected_mean'] = (perc_diff > 0.20)
    
    return results

# ==================================================================================
# 3. ALERT RULE & ORCHESTRATION
# ==================================================================================

def run_monitoring_pipeline(batch_name, baseline_df, current_df):
    print(f"\n--- MONITORING REPORT: {batch_name} ---")
    
    # 1. Run Data Quality
    dq_res = check_data_quality(current_df)
    print(f"[DQ] Null Check: {'PASS' if dq_res['null_check_pass'] else 'FAIL'}")
    print(f"[DQ] Range Check: {'PASS' if dq_res['range_check_pass'] else 'FAIL'}")
    
    if not (dq_res['null_check_pass'] and dq_res['range_check_pass']):
        print("[CRITICAL] Data Quality Failed. Halting pipeline.")
        return

    # 2. Run Drift Checks
    drift_res = check_drift(baseline_df, current_df)
    print(f"[DRIFT] KS Test P-Value: {drift_res['ks_p_value']:.5f} (Detected: {drift_res['drift_detected_ks']})")
    print(f"[DRIFT] Mean Shift: {drift_res['mean_diff_perc']:.2%} (Detected: {drift_res['drift_detected_mean']})")
    
    # 3. Alert Rule (Plain Language)
    # (iii) Alert Rule:
    # "IF KS-Test P-Value < 0.05 OR Mean Shift > 20%, THEN Trigger Retraining Alert."
    
    if drift_res['drift_detected_ks'] or drift_res['drift_detected_mean']:
        print("\n>>> 🚨 ALERT FIRED 🚨 <<<")
        print("REASON: Significant distribution shift detected in 'income' feature.")
        print("ACTION: 1. Verify data source health (upstream check).")
        print("        2. If data is valid, trigger Model Retraining Pipeline on new data.")
    else:
        print("\n>>> ✅ SYSTEM HEALTHY ✅ <<<")
        print("No significant drift detected. Continue normal operation.")

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

if __name__ == '__main__':
    # 1. Generate Baseline
    print("Generating Baseline Data...")
    df_baseline = generate_data('baseline')
    
    # 2. Generate Batches
    print("Generating Production Batches...")
    df_batch1_drift = generate_data('data_drift')    # High Income
    df_batch2_concept = generate_data('concept_drift') # Policy Change
    
    # 3. Run Pipeline on Self (Sanity Check) -> Should pass
    run_monitoring_pipeline("Baseline Test (Self)", df_baseline, df_baseline)
    
    # 4. Run Pipeline on Batch 1 (Data Drift)
    # Expect: Drift Alert (Income distribution changed)
    run_monitoring_pipeline("Batch 1 (Data Drift)", df_baseline, df_batch1_drift)
    
    # 5. Run Pipeline on Batch 2 (Concept Drift)
    # NOTE: Standard Distribution Checks (KS/Mean) on Feature X might MISS Concept Drift 
    # if P(X) is same but P(Y|X) changed!
    # Let's see if our checks catch it.
    # In our simulation, 'concept_drift' batch resets income to normal mean=50k.
    # So P(X) is same as baseline. KS test should PASS.
    # This demonstrates why we need Ground Truth monitoring (Check 3 - performance) separately.
    run_monitoring_pipeline("Batch 2 (Concept Drift - Hidden?)", df_baseline, df_batch2_concept)
    
    print("\n[OBSERVATION ON BATCH 2]:")
    print("Did you notice Batch 2 passed the X-Distribution checks?")
    print("This is accurate! Concept Drift (P(Y|X) change) is NOT Data Drift (P(X) change).")
    print("To catch Batch 2, we would need to monitor Model Performance (Accuracy/loss).")
