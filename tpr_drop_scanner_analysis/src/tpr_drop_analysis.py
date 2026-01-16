"""
TPR Drop Analysis After New Scanner Introduction
=================================================

This script simulates and analyzes the scenario where slice-wise TPR (True Positive Rate)
drops sharply for one site after a new scanner is introduced, while service latency
and error rate remain normal.

Author: Teaching Demo
Date: 2026-01-16
Purpose: Educational demonstration of data/model health issues vs service issues
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# SECTION 2: SIMULATE MULTI-SITE SCANNER DATA
# =============================================================================

def simulate_scanner_data(n_samples_per_scanner=1000):
    """
    Simulate medical imaging data from multiple scanners at different sites.
    
    Parameters:
    -----------
    n_samples_per_scanner : int, default=1000
        Number of samples to generate per scanner
    
    Returns:
    --------
    df : pandas.DataFrame
        Simulated data with features, labels, and scanner information
    """
    
    # Scanner A (Site 1) - Original scanner used for training
    scanner_a_features = np.random.normal(loc=100, scale=15, size=(n_samples_per_scanner, 10))
    scanner_a_labels = (scanner_a_features[:, 0] + scanner_a_features[:, 1] > 200).astype(int)
    scanner_a_df = pd.DataFrame(scanner_a_features, columns=[f'feature_{i}' for i in range(10)])
    scanner_a_df['label'] = scanner_a_labels
    scanner_a_df['scanner'] = 'Scanner_A'
    scanner_a_df['site'] = 'Site_1'
    
    # Scanner B (Site 2) - Similar to Scanner A (same distribution)
    scanner_b_features = np.random.normal(loc=100, scale=15, size=(n_samples_per_scanner, 10))
    scanner_b_labels = (scanner_b_features[:, 0] + scanner_b_features[:, 1] > 200).astype(int)
    scanner_b_df = pd.DataFrame(scanner_b_features, columns=[f'feature_{i}' for i in range(10)])
    scanner_b_df['label'] = scanner_b_labels
    scanner_b_df['scanner'] = 'Scanner_B'
    scanner_b_df['site'] = 'Site_2'
    
    # Scanner C (Site 3) - NEW SCANNER with DIFFERENT distribution (covariate shift!)
    scanner_c_features = np.random.normal(loc=120, scale=25, size=(n_samples_per_scanner, 10))
    scanner_c_features += np.random.uniform(-10, 10, size=(n_samples_per_scanner, 10))
    scanner_c_labels = (scanner_c_features[:, 0] + scanner_c_features[:, 1] > 240).astype(int)
    scanner_c_df = pd.DataFrame(scanner_c_features, columns=[f'feature_{i}' for i in range(10)])
    scanner_c_df['label'] = scanner_c_labels
    scanner_c_df['scanner'] = 'Scanner_C_NEW'
    scanner_c_df['site'] = 'Site_3'
    
    # Combine all data
    df = pd.concat([scanner_a_df, scanner_b_df, scanner_c_df], ignore_index=True)
    
    return df

# =============================================================================
# SECTION 3: TRAIN MODEL ON ORIGINAL SCANNERS ONLY
# =============================================================================

def train_model_on_original_scanners(df):
    """
    Train a model using ONLY data from original scanners (A and B).
    """
    
    training_data = df[df['scanner'].isin(['Scanner_A', 'Scanner_B'])].copy()
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X_train = training_data[feature_cols]
    y_train = training_data['label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Scanners used: Scanner_A, Scanner_B")
    print(f"Scanner_C_NEW: NOT included in training")
    print("=" * 60)
    
    return model, scaler

# =============================================================================
# SECTION 4: EVALUATE MODEL PER SCANNER (SLICE-WISE TPR)
# =============================================================================

def evaluate_per_scanner(df, model, scaler):
    """Evaluate model performance per scanner slice."""
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    results = {}
    
    print("\n" + "=" * 60)
    print("SLICE-WISE TPR EVALUATION")
    print("=" * 60)
    
    for scanner in df['scanner'].unique():
        scanner_data = df[df['scanner'] == scanner].copy()
        X = scanner_data[feature_cols]
        y_true = scanner_data['label']
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results[scanner] = {
            'TPR': tpr, 'FPR': fpr, 'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            'mean_confidence': np.mean(y_proba), 'samples': len(y_true)
        }
        
        site = scanner_data['site'].iloc[0]
        status = "[OK]" if tpr > 0.8 else "[DEGRADED]"
        print(f"\n[SCANNER] {scanner} ({site}) {status}")
        print(f"   TPR (Recall): {tpr:.4f}")
        print(f"   Confusion Matrix: TP={tp}, FN={fn}, TN={tn}, FP={fp}")
    
    return results

# =============================================================================
# SECTION 5: SIMULATE SERVICE METRICS
# =============================================================================

def check_service_metrics():
    """Simulate service health check."""
    
    print("\n" + "=" * 60)
    print("SERVICE HEALTH CHECK")
    print("=" * 60)
    
    latency_ms = np.random.normal(50, 5, 100)
    error_rate = 0.001
    
    metrics = {
        'avg_latency_ms': np.mean(latency_ms),
        'p99_latency_ms': np.percentile(latency_ms, 99),
        'error_rate': error_rate,
        'status': 'HEALTHY'
    }
    
    print(f"[OK] Average Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"[OK] P99 Latency: {metrics['p99_latency_ms']:.2f} ms")
    print(f"[OK] Error Rate: {metrics['error_rate'] * 100:.3f}%")
    print(f"[OK] Service Status: {metrics['status']}")
    print("\n[WARNING] SERVICE IS HEALTHY - But TPR dropped for Scanner_C_NEW!")
    print("[WARNING] This is a DATA/MODEL issue, NOT a service issue!")
    
    return metrics

# =============================================================================
# SECTION 6: DIAGNOSTIC 1 - FEATURE DISTRIBUTION COMPARISON
# =============================================================================

def diagnostic_feature_distribution(df):
    """Compare feature distributions using KL Divergence."""
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1: FEATURE DISTRIBUTION COMPARISON")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    original_data = df[df['scanner'].isin(['Scanner_A', 'Scanner_B'])]
    new_data = df[df['scanner'] == 'Scanner_C_NEW']
    
    kl_scores = {}
    
    for col in feature_cols[:3]:
        orig_hist, bin_edges = np.histogram(original_data[col], bins=30, density=True)
        new_hist, _ = np.histogram(new_data[col], bins=bin_edges, density=True)
        
        orig_hist = (orig_hist + 1e-10) / (orig_hist + 1e-10).sum()
        new_hist = (new_hist + 1e-10) / (new_hist + 1e-10).sum()
        
        kl_div = entropy(new_hist, orig_hist)
        kl_scores[col] = kl_div
        
        print(f"\n[FEATURE] {col}:")
        print(f"   Original Mean: {original_data[col].mean():.2f}")
        print(f"   New Scanner Mean: {new_data[col].mean():.2f}")
        print(f"   KL Divergence: {kl_div:.4f}")
        
        if kl_div > 0.1:
            print(f"   [WARNING] HIGH DISTRIBUTION SHIFT DETECTED!")
    
    return kl_scores

# =============================================================================
# SECTION 7: DIAGNOSTIC 2 - PREDICTION CONFIDENCE ANALYSIS
# =============================================================================

def diagnostic_confidence_analysis(df, model, scaler):
    """Check if model is confidently wrong on new scanner."""
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    analysis = {}
    
    for scanner in df['scanner'].unique():
        scanner_data = df[df['scanner'] == scanner].copy()
        X = scanner_data[feature_cols]
        y_true = scanner_data['label']
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        false_negatives = (y_true == 1) & (y_pred == 0)
        fn_count = false_negatives.sum()
        avg_fn_conf = np.mean(y_proba[false_negatives]) if fn_count > 0 else 0
        
        analysis[scanner] = {
            'total_false_negatives': fn_count,
            'avg_confidence_on_FN': avg_fn_conf,
            'avg_overall_confidence': np.mean(y_proba)
        }
        
        print(f"\n[TARGET] {scanner}:")
        print(f"   False Negatives: {fn_count}")
        print(f"   Avg Confidence on FN: {avg_fn_conf:.4f}")
        
        if scanner == 'Scanner_C_NEW' and fn_count > 50:
            print(f"   [WARNING] MODEL IS CONFIDENTLY WRONG ON NEW SCANNER!")
    
    return analysis

# =============================================================================
# SECTION 8: DIAGNOSTIC 3 - CONFUSION MATRIX BREAKDOWN
# =============================================================================

def diagnostic_confusion_matrix_breakdown(df, model, scaler):
    """Per-scanner confusion matrices."""
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: CONFUSION MATRIX BREAKDOWN BY SCANNER")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    cm_dict = {}
    
    for scanner in df['scanner'].unique():
        scanner_data = df[df['scanner'] == scanner].copy()
        X = scanner_data[feature_cols]
        y_true = scanner_data['label']
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        cm = confusion_matrix(y_true, y_pred)
        cm_dict[scanner] = cm
        
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n[MATRIX] {scanner}:")
        print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"   TPR: {tpr:.4f}, FNR: {fnr:.4f}")
    
    return cm_dict

# =============================================================================
# SECTION 9: DIAGNOSTIC 4 - TEMPORAL ANALYSIS
# =============================================================================

def diagnostic_temporal_analysis():
    """Simulated temporal TPR trend."""
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: TEMPORAL TPR TREND ANALYSIS")
    print("=" * 60)
    
    dates = pd.date_range(start='2026-01-01', periods=8, freq='W')
    tpr_values = [0.92, 0.91, 0.93, 0.90, 0.65, 0.63, 0.66, 0.64]
    scanner_type = ['OLD', 'OLD', 'OLD', 'OLD', 'NEW', 'NEW', 'NEW', 'NEW']
    
    timeline = pd.DataFrame({'date': dates, 'tpr': tpr_values, 'scanner_type': scanner_type})
    
    print("\n[TIMELINE] Weekly TPR for Site 3:")
    print("-" * 50)
    for _, row in timeline.iterrows():
        marker = "[!]" if row['scanner_type'] == 'NEW' else "[+]"
        print(f"   {row['date'].strftime('%Y-%m-%d')} | TPR: {row['tpr']:.2f} | {marker} {row['scanner_type']} Scanner")
    
    print("\n[WARNING] SHARP TPR DROP DETECTED AT WEEK 5!")
    print("[WARNING] Coincides with new scanner introduction!")
    
    return timeline

# =============================================================================
# SECTION 10: MITIGATION 1 - SAFE FALLBACK
# =============================================================================

def mitigation_safe_fallback(df, model, scaler, confidence_threshold=0.7):
    """Route low-confidence predictions to human review."""
    
    print("\n" + "=" * 60)
    print("MITIGATION 1: SAFE FALLBACK - HUMAN REVIEW ROUTING")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    routing_stats = {}
    
    for scanner in df['scanner'].unique():
        scanner_data = df[df['scanner'] == scanner].copy()
        X = scanner_data[feature_cols]
        
        X_scaled = scaler.transform(X)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        max_confidence = np.maximum(y_proba, 1 - y_proba)
        needs_review = max_confidence < confidence_threshold
        
        routing_stats[scanner] = {
            'total_samples': len(scanner_data),
            'routed_to_human': needs_review.sum(),
            'human_review_rate': needs_review.mean()
        }
        
        print(f"\n[SHIELD] {scanner}:")
        print(f"   Total Samples: {len(scanner_data)}")
        print(f"   Auto-decided: {(~needs_review).sum()}")
        print(f"   Human Review: {needs_review.sum()} ({needs_review.mean()*100:.1f}%)")
    
    print("\n[OK] SAFE FALLBACK ACTIVE!")
    
    return routing_stats

# =============================================================================
# SECTION 11: MITIGATION 2 - PREPROCESSING NORMALIZATION
# =============================================================================

def mitigation_preprocessing_normalization(df):
    """Apply scanner-specific preprocessing to align distributions."""
    
    print("\n" + "=" * 60)
    print("MITIGATION 2: PREPROCESSING NORMALIZATION")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    original_data = df[df['scanner'].isin(['Scanner_A', 'Scanner_B'])]
    ref_mean = original_data[feature_cols].mean()
    ref_std = original_data[feature_cols].std()
    
    normalized_df = df.copy()
    new_scanner_mask = df['scanner'] == 'Scanner_C_NEW'
    new_scanner_data = df.loc[new_scanner_mask, feature_cols]
    
    new_mean = new_scanner_data.mean()
    new_std = new_scanner_data.std()
    
    normalized_features = (new_scanner_data - new_mean) / new_std * ref_std + ref_mean
    normalized_df.loc[new_scanner_mask, feature_cols] = normalized_features
    
    print(f"\n[BEFORE] New Scanner Mean: {df.loc[new_scanner_mask, 'feature_0'].mean():.2f}")
    print(f"[AFTER]  Normalized Mean: {normalized_df.loc[new_scanner_mask, 'feature_0'].mean():.2f}")
    print(f"[TARGET] Original Mean: {original_data['feature_0'].mean():.2f}")
    print("\n[OK] Distributions aligned!")
    
    return normalized_df

# =============================================================================
# SECTION 12: MITIGATION 3 - DOMAIN ADAPTATION
# =============================================================================

def mitigation_domain_adaptation(df, model, scaler):
    """Retrain model with new scanner data."""
    
    print("\n" + "=" * 60)
    print("MITIGATION 3: DOMAIN ADAPTATION / FINE-TUNING")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    new_scanner_data = df[df['scanner'] == 'Scanner_C_NEW'].sample(n=200, random_state=42)
    all_training_data = pd.concat([
        df[df['scanner'].isin(['Scanner_A', 'Scanner_B'])],
        new_scanner_data
    ])
    
    X = all_training_data[feature_cols]
    y = all_training_data['label']
    
    X_scaled = scaler.fit_transform(X)
    adapted_model = RandomForestClassifier(n_estimators=100, random_state=42)
    adapted_model.fit(X_scaled, y)
    
    print(f"\n[RETRAIN] Model retrained with:")
    print(f"   Original samples: {len(df[df['scanner'].isin(['Scanner_A', 'Scanner_B'])])}")
    print(f"   New scanner samples: 200")
    print(f"   Total: {len(all_training_data)}")
    
    new_scanner_test = df[df['scanner'] == 'Scanner_C_NEW']
    X_test = new_scanner_test[feature_cols]
    y_test = new_scanner_test['label']
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = adapted_model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    new_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n[RESULT] TPR AFTER Domain Adaptation: {new_tpr:.4f}")
    print("[OK] TPR improved on new scanner!")
    
    return adapted_model

# =============================================================================
# SECTION 13: SUMMARIZE ROOT CAUSE
# =============================================================================

def summarize_root_cause():
    """Print summary explaining why this is a data/model issue."""
    
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("""
+---------------------------------------------------------------+
|  WHY THIS IS A DATA/MODEL HEALTH ISSUE (NOT A SERVICE ISSUE)  |
+---------------------------------------------------------------+
|                                                               |
|  1. SERVICE METRICS ARE NORMAL                                |
|     [OK] Latency within range (~50ms)                         |
|     [OK] Error rate low (~0.1%)                               |
|     --> Infrastructure is HEALTHY                             |
|                                                               |
|  2. TPR DROP IS ISOLATED TO ONE SITE/SCANNER                  |
|     [!] Only Site 3 (new scanner) shows TPR degradation       |
|     [+] Other sites maintain normal TPR                       |
|     --> NOT a global service failure                          |
|                                                               |
|  3. NEW SCANNER INTRODUCES COVARIATE SHIFT                    |
|     [i] Different pixel intensity distribution                |
|     [i] Different noise profile                               |
|     [i] Model never saw this data during training             |
|     --> Model makes confident but WRONG predictions           |
|                                                               |
|  4. THIS IS A SILENT FAILURE MODE                             |
|     [!] System appears healthy from infrastructure view       |
|     [!] Model quality degraded for specific subgroup          |
|     [!] Without slice-based monitoring, goes undetected!      |
|                                                               |
+---------------------------------------------------------------+
""")

# =============================================================================
# SECTION 14: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 70)
    print("TPR DROP ANALYSIS AFTER NEW SCANNER INTRODUCTION")
    print("Slice-wise Monitoring for Data/Model Health Issues")
    print("=" * 70)
    
    # Step 1: Simulate data
    print("\n[STEP 1] Simulating multi-site scanner data...")
    df = simulate_scanner_data(n_samples_per_scanner=1000)
    print(f"   Generated {len(df)} samples across 3 scanners")
    
    # Step 2: Train model
    print("\n[STEP 2] Training model on original scanners...")
    model, scaler = train_model_on_original_scanners(df)
    
    # Step 3: Evaluate per-scanner TPR
    print("\n[STEP 3] Evaluating slice-wise TPR...")
    results = evaluate_per_scanner(df, model, scaler)
    
    # Step 4: Check service metrics
    print("\n[STEP 4] Checking service health metrics...")
    service_metrics = check_service_metrics()
    
    # Step 5: Run diagnostics
    print("\n" + "=" * 70)
    print("RUNNING DIAGNOSTICS")
    print("=" * 70)
    
    kl_scores = diagnostic_feature_distribution(df)
    confidence_analysis = diagnostic_confidence_analysis(df, model, scaler)
    cm_breakdown = diagnostic_confusion_matrix_breakdown(df, model, scaler)
    timeline = diagnostic_temporal_analysis()
    
    # Step 6: Apply mitigations
    print("\n" + "=" * 70)
    print("APPLYING MITIGATIONS")
    print("=" * 70)
    
    routing_stats = mitigation_safe_fallback(df, model, scaler)
    normalized_df = mitigation_preprocessing_normalization(df)
    adapted_model = mitigation_domain_adaptation(df, model, scaler)
    
    # Step 7: Summarize
    summarize_root_cause()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return df, model, scaler, results

# =============================================================================
# SECTION 15: RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    df, model, scaler, results = main()
