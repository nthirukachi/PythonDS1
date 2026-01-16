"""
================================================================================
CANARY ROLLOUT PREDICTION SHIFT ANALYSIS
================================================================================
This script demonstrates how to detect and analyze prediction distribution shifts
during a canary deployment of a machine learning classifier model.

SCENARIO:
- A new classifier model is deployed using canary rollout
- After 2 hours, Class A predictions shift from 20% to 55%
- Latency and error rates are normal
- Ground truth labels are NOT available yet

OBJECTIVES:
1. Identify plausible causes for the shift
2. Run diagnostic checks (data quality, input drift, prediction behavior)
3. Determine the safest next action (continue, pause, rollback, or route to review)

Author: Teaching Demo
Date: 2026-01-16
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ==============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# SECTION 2: SIMULATE BASELINE (OLD) MODEL PREDICTIONS
# ==============================================================================

def simulate_baseline_predictions(n_samples=10000):
    """
    Simulate predictions from the BASELINE (old) model.
    
    The baseline model has the following class distribution:
    - Class A: 20%
    - Class B: 50%
    - Class C: 30%
    
    Parameters:
    -----------
    n_samples : int
        Number of prediction samples to generate
        
    Returns:
    --------
    numpy.ndarray
        Array of class predictions (0=A, 1=B, 2=C)
    """
    # Define baseline class probabilities
    baseline_probs = [0.20, 0.50, 0.30]  # A=20%, B=50%, C=30%
    
    # Generate predictions based on these probabilities
    predictions = np.random.choice(
        [0, 1, 2],           # Class labels: 0=A, 1=B, 2=C
        size=n_samples,      # Number of samples
        p=baseline_probs     # Probability of each class
    )
    
    return predictions


def simulate_baseline_features(n_samples=10000):
    """
    Simulate input features for baseline model.
    
    Features:
    - feature_1: Normally distributed, mean=50, std=10
    - feature_2: Normally distributed, mean=100, std=20
    - feature_3: Uniformly distributed between 0 and 1
    
    Parameters:
    -----------
    n_samples : int
        Number of feature samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing simulated features
    """
    features = pd.DataFrame({
        'feature_1': np.random.normal(loc=50, scale=10, size=n_samples),
        'feature_2': np.random.normal(loc=100, scale=20, size=n_samples),
        'feature_3': np.random.uniform(low=0, high=1, size=n_samples)
    })
    
    return features


# ==============================================================================
# SECTION 3: SIMULATE CANARY (NEW) MODEL PREDICTIONS WITH DRIFT
# ==============================================================================

def simulate_canary_predictions(n_samples=10000, drift_type='distribution_shift'):
    """
    Simulate predictions from the CANARY (new) model with prediction drift.
    
    The canary model exhibits a SHIFTED class distribution:
    - Class A: 55% (was 20% - MAJOR INCREASE!)
    - Class B: 30% (was 50%)
    - Class C: 15% (was 30%)
    
    Parameters:
    -----------
    n_samples : int
        Number of prediction samples to generate
    drift_type : str
        Type of drift: 'distribution_shift', 'calibration_issue', 'feature_drift'
        
    Returns:
    --------
    numpy.ndarray
        Array of class predictions (0=A, 1=B, 2=C)
    """
    if drift_type == 'distribution_shift':
        # Shifted class probabilities (as observed in the scenario)
        canary_probs = [0.55, 0.30, 0.15]  # A=55%, B=30%, C=15%
    elif drift_type == 'calibration_issue':
        # Model is over-confident about Class A
        canary_probs = [0.60, 0.25, 0.15]
    else:
        # Slight drift
        canary_probs = [0.40, 0.40, 0.20]
    
    # Generate predictions based on shifted probabilities
    predictions = np.random.choice(
        [0, 1, 2],           # Class labels: 0=A, 1=B, 2=C
        size=n_samples,
        p=canary_probs
    )
    
    return predictions


def simulate_canary_features(n_samples=10000, drift_present=True):
    """
    Simulate input features for canary model (potentially with drift).
    
    If drift_present=True:
    - feature_1: Mean shifts from 50 to 65 (COVARIATE SHIFT!)
    - feature_2: Mean remains at 100
    - feature_3: Remains uniform
    
    Parameters:
    -----------
    n_samples : int
        Number of feature samples to generate
    drift_present : bool
        Whether to simulate input drift
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing simulated features
    """
    if drift_present:
        # DRIFTED features - feature_1 mean has shifted!
        features = pd.DataFrame({
            'feature_1': np.random.normal(loc=65, scale=10, size=n_samples),  # Mean shifted!
            'feature_2': np.random.normal(loc=100, scale=20, size=n_samples),
            'feature_3': np.random.uniform(low=0, high=1, size=n_samples)
        })
    else:
        # No drift - same as baseline
        features = pd.DataFrame({
            'feature_1': np.random.normal(loc=50, scale=10, size=n_samples),
            'feature_2': np.random.normal(loc=100, scale=20, size=n_samples),
            'feature_3': np.random.uniform(low=0, high=1, size=n_samples)
        })
    
    return features


# ==============================================================================
# SECTION 4: DATA QUALITY CHECKS
# ==============================================================================

def check_data_quality(df, check_name="Data"):
    """
    Perform data quality checks on input features.
    
    Checks performed:
    1. Missing values count
    2. Data type validation
    3. Out-of-range values detection
    4. Basic statistics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    check_name : str
        Name for the check (for logging)
        
    Returns:
    --------
    dict
        Dictionary containing quality check results
    """
    print(f"\n{'='*60}")
    print(f"DATA QUALITY CHECK: {check_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Check 1: Missing Values
    missing = df.isnull().sum()
    results['missing_values'] = missing.to_dict()
    print(f"\n[CHECK 1] Missing Values:")
    print(f"  Total missing: {missing.sum()}")
    for col, count in missing.items():
        print(f"    {col}: {count} missing")
    
    # Check 2: Data Types
    results['data_types'] = df.dtypes.astype(str).to_dict()
    print(f"\n[CHECK 2] Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col}: {dtype}")
    
    # Check 3: Basic Statistics
    print(f"\n[CHECK 3] Basic Statistics:")
    print(df.describe().round(2).to_string())
    
    # Check 4: Out-of-range detection (example: negative values where unexpected)
    results['negative_values'] = {}
    print(f"\n[CHECK 4] Negative Value Check:")
    for col in df.select_dtypes(include=[np.number]).columns:
        neg_count = (df[col] < 0).sum()
        results['negative_values'][col] = neg_count
        status = "OK" if neg_count == 0 else f"WARNING: {neg_count} negative values!"
        print(f"    {col}: {status}")
    
    return results


# ==============================================================================
# SECTION 5: INPUT DRIFT DETECTION (COVARIATE SHIFT)
# ==============================================================================

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) to detect distribution shift.
    
    PSI INTERPRETATION:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift (investigate)
    - PSI >= 0.25: Significant shift (action required!)
    
    Parameters:
    -----------
    expected : array-like
        Expected distribution (baseline)
    actual : array-like
        Actual distribution (canary)
    buckets : int
        Number of buckets for discretization
        
    Returns:
    --------
    float
        PSI value
    """
    # Create buckets based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    # Calculate frequencies for each bucket
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Convert to proportions
    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_prop = np.where(expected_prop == 0, 0.0001, expected_prop)
    actual_prop = np.where(actual_prop == 0, 0.0001, actual_prop)
    
    # Calculate PSI
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    
    return psi


def detect_input_drift(baseline_features, canary_features):
    """
    Detect input drift between baseline and canary feature distributions.
    
    Uses two methods:
    1. Kolmogorov-Smirnov (KS) Test - detects distribution differences
    2. Population Stability Index (PSI) - quantifies distribution shift
    
    Parameters:
    -----------
    baseline_features : pandas.DataFrame
        Features from baseline period
    canary_features : pandas.DataFrame
        Features from canary period
        
    Returns:
    --------
    dict
        Dictionary containing drift detection results
    """
    print(f"\n{'='*60}")
    print("INPUT DRIFT DETECTION (COVARIATE SHIFT)")
    print(f"{'='*60}")
    
    results = {}
    
    for column in baseline_features.columns:
        baseline_data = baseline_features[column].values
        canary_data = canary_features[column].values
        
        # KS Test
        ks_statistic, ks_pvalue = stats.ks_2samp(baseline_data, canary_data)
        
        # PSI
        psi_value = calculate_psi(baseline_data, canary_data)
        
        # Determine drift status
        if psi_value >= 0.25:
            drift_status = "SIGNIFICANT DRIFT!"
        elif psi_value >= 0.1:
            drift_status = "MODERATE DRIFT"
        else:
            drift_status = "No significant drift"
        
        results[column] = {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'psi': psi_value,
            'drift_status': drift_status
        }
        
        print(f"\n[{column}]")
        print(f"  KS Statistic: {ks_statistic:.4f}")
        print(f"  KS P-Value: {ks_pvalue:.4f}")
        print(f"  PSI: {psi_value:.4f}")
        print(f"  Status: {drift_status}")
    
    return results


# ==============================================================================
# SECTION 6: PREDICTION BEHAVIOR ANALYSIS
# ==============================================================================

def analyze_prediction_behavior(baseline_preds, canary_preds):
    """
    Analyze and compare prediction behavior between baseline and canary models.
    
    Metrics computed:
    1. Class distribution comparison
    2. Absolute shift per class
    3. Chi-square test for distribution difference
    
    Parameters:
    -----------
    baseline_preds : numpy.ndarray
        Predictions from baseline model
    canary_preds : numpy.ndarray
        Predictions from canary model
        
    Returns:
    --------
    dict
        Dictionary containing prediction behavior analysis
    """
    print(f"\n{'='*60}")
    print("PREDICTION BEHAVIOR ANALYSIS")
    print(f"{'='*60}")
    
    # Class labels
    class_names = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    
    # Calculate distributions
    baseline_counts = Counter(baseline_preds)
    canary_counts = Counter(canary_preds)
    
    n_baseline = len(baseline_preds)
    n_canary = len(canary_preds)
    
    print("\n[DISTRIBUTION COMPARISON]")
    print(f"{'Class':<12} {'Baseline':<15} {'Canary':<15} {'Shift':<15}")
    print("-" * 57)
    
    results = {}
    
    for class_id in [0, 1, 2]:
        baseline_pct = (baseline_counts.get(class_id, 0) / n_baseline) * 100
        canary_pct = (canary_counts.get(class_id, 0) / n_canary) * 100
        shift = canary_pct - baseline_pct
        
        shift_indicator = ""
        if abs(shift) >= 20:
            shift_indicator = "MAJOR SHIFT!"
        elif abs(shift) >= 10:
            shift_indicator = "Notable shift"
        
        results[class_names[class_id]] = {
            'baseline_pct': baseline_pct,
            'canary_pct': canary_pct,
            'shift': shift,
            'alert': shift_indicator
        }
        
        print(f"{class_names[class_id]:<12} {baseline_pct:>6.1f}%{'':<8} {canary_pct:>6.1f}%{'':<8} {shift:>+6.1f}%  {shift_indicator}")
    
    # Chi-square test
    baseline_freq = [baseline_counts.get(i, 0) for i in [0, 1, 2]]
    canary_freq = [canary_counts.get(i, 0) for i in [0, 1, 2]]
    
    # Normalize frequencies for chi-square
    expected_freq = [f * (n_canary / n_baseline) for f in baseline_freq]
    chi2_stat, chi2_pvalue = stats.chisquare(canary_freq, f_exp=expected_freq)
    
    print(f"\n[CHI-SQUARE TEST]")
    print(f"  Chi-square statistic: {chi2_stat:.2f}")
    print(f"  P-value: {chi2_pvalue:.6f}")
    
    if chi2_pvalue < 0.05:
        print(f"  Result: SIGNIFICANT difference in distributions (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    results['chi_square'] = {
        'statistic': chi2_stat,
        'pvalue': chi2_pvalue,
        'significant': chi2_pvalue < 0.05
    }
    
    return results


# ==============================================================================
# SECTION 7: DECISION LOGIC - SAFEST NEXT ACTION
# ==============================================================================

def determine_safest_action(data_quality_results, drift_results, prediction_results):
    """
    Determine the safest next action based on all diagnostic checks.
    
    Actions possible:
    1. CONTINUE: All checks pass, proceed with rollout
    2. PAUSE: Minor concerns, hold and monitor
    3. ROLLBACK: Significant issues detected, revert to baseline
    4. ROUTE TO REVIEW: Uncertain, needs human review
    
    Parameters:
    -----------
    data_quality_results : dict
        Results from data quality checks
    drift_results : dict
        Results from input drift detection
    prediction_results : dict
        Results from prediction behavior analysis
        
    Returns:
    --------
    tuple
        (action, justification, confidence)
    """
    print(f"\n{'='*60}")
    print("DECISION: SAFEST NEXT ACTION")
    print(f"{'='*60}")
    
    # Collect risk signals
    risk_signals = []
    
    # Check 1: Data quality issues
    total_missing = sum(data_quality_results.get('missing_values', {}).values())
    if total_missing > 0:
        risk_signals.append(f"Data quality: {total_missing} missing values")
    
    # Check 2: Input drift
    significant_drift = False
    for feature, results in drift_results.items():
        if results.get('psi', 0) >= 0.25:
            significant_drift = True
            risk_signals.append(f"Input drift: {feature} has PSI={results['psi']:.3f}")
    
    # Check 3: Prediction shift
    major_shift = False
    for class_name, results in prediction_results.items():
        if isinstance(results, dict) and 'shift' in results:
            if abs(results['shift']) >= 20:
                major_shift = True
                risk_signals.append(f"Prediction shift: {class_name} shifted by {results['shift']:+.1f}%")
    
    # Decision logic
    print("\n[RISK SIGNALS DETECTED]")
    if not risk_signals:
        print("  No significant risk signals detected.")
    else:
        for signal in risk_signals:
            print(f"  - {signal}")
    
    print("\n[DECISION ANALYSIS]")
    
    # Determine action
    if major_shift and significant_drift:
        action = "ROLLBACK"
        confidence = 0.85
        justification = """
        JUSTIFICATION FOR ROLLBACK:
        1. Significant input drift detected (PSI >= 0.25)
        2. Major prediction distribution shift (>= 20% for a class)
        3. These combined signals indicate the model is receiving different
           data than it was trained on, causing unreliable predictions.
        4. Without ground truth labels, we cannot verify model accuracy.
        5. The safest action is to rollback to the stable baseline model
           while investigating the root cause.
        """
    elif major_shift and not significant_drift:
        action = "PAUSE + ROUTE TO REVIEW"
        confidence = 0.75
        justification = """
        JUSTIFICATION FOR PAUSE + ROUTE TO REVIEW:
        1. Major prediction shift detected, but NO input drift.
        2. This suggests the issue is with the MODEL itself, not the data.
        3. Possible causes:
           - Model calibration difference
           - Training data distribution mismatch
           - Feature preprocessing discrepancy
        4. Human review needed to determine if the new predictions
           are actually BETTER or WORSE than baseline.
        5. Pause deployment and route to ML team for investigation.
        """
    elif significant_drift and not major_shift:
        action = "PAUSE"
        confidence = 0.70
        justification = """
        JUSTIFICATION FOR PAUSE:
        1. Input drift detected, but predictions are relatively stable.
        2. The model might be handling the drift well, but this is risky.
        3. Hold the canary at current traffic level.
        4. Monitor for the next few hours before proceeding.
        5. If drift continues, consider rollback.
        """
    else:
        action = "CONTINUE"
        confidence = 0.60
        justification = """
        JUSTIFICATION FOR CONTINUE:
        1. No significant input drift detected.
        2. Prediction distribution is within acceptable bounds.
        3. Data quality checks passed.
        4. Continue the canary rollout cautiously.
        5. Increase monitoring frequency until labels are available.
        """
    
    print(f"\n[RECOMMENDED ACTION]")
    print(f"  Action: {action}")
    print(f"  Confidence: {confidence*100:.0f}%")
    print(f"\n{justification}")
    
    return action, justification, confidence


# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function to run the complete canary rollout prediction shift analysis.
    """
    print("=" * 70)
    print("CANARY ROLLOUT PREDICTION SHIFT ANALYSIS")
    print("=" * 70)
    print("\nSCENARIO:")
    print("- New classifier deployed via canary rollout")
    print("- After 2 hours: Class A predictions shifted from 20% to 55%")
    print("- Latency: Normal | Error Rate: Normal | Labels: Not available")
    print("=" * 70)
    
    # Step 1: Simulate baseline data
    print("\n[STEP 1] Simulating baseline (old) model data...")
    baseline_predictions = simulate_baseline_predictions(n_samples=10000)
    baseline_features = simulate_baseline_features(n_samples=10000)
    
    # Step 2: Simulate canary data with drift
    print("[STEP 2] Simulating canary (new) model data with drift...")
    canary_predictions = simulate_canary_predictions(n_samples=10000)
    canary_features = simulate_canary_features(n_samples=10000, drift_present=True)
    
    # Step 3: Data quality checks
    print("\n[STEP 3] Running data quality checks...")
    baseline_quality = check_data_quality(baseline_features, "Baseline Features")
    canary_quality = check_data_quality(canary_features, "Canary Features")
    
    # Step 4: Input drift detection
    print("\n[STEP 4] Detecting input drift (covariate shift)...")
    drift_results = detect_input_drift(baseline_features, canary_features)
    
    # Step 5: Prediction behavior analysis
    print("\n[STEP 5] Analyzing prediction behavior...")
    prediction_results = analyze_prediction_behavior(baseline_predictions, canary_predictions)
    
    # Step 6: Determine safest action
    print("\n[STEP 6] Determining safest next action...")
    action, justification, confidence = determine_safest_action(
        canary_quality, 
        drift_results, 
        prediction_results
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nFINAL RECOMMENDATION: {action}")
    print(f"CONFIDENCE LEVEL: {confidence*100:.0f}%")
    print("\nKEY FINDINGS:")
    print("1. Class A predictions shifted from 20% → 55% (35 percentage point increase)")
    print("2. Input feature 'feature_1' shows significant covariate shift (PSI > 0.25)")
    print("3. Model behavior suggests potential data pipeline or preprocessing issue")
    print("\nNEXT STEPS:")
    print("- Investigate upstream data sources for changes")
    print("- Compare feature engineering pipelines between training and serving")
    print("- Wait for ground truth labels to measure actual accuracy impact")
    print("=" * 70)


# ==============================================================================
# SECTION 9: SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
