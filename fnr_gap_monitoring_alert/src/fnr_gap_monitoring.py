# ============================================================================
# FNR Gap Monitoring Alert System
# ============================================================================
# This script implements a monitoring system to detect unequal harm across
# operational slices in deployed binary classifiers using FNR (False Negative Rate).
#
# Key Formula:
#   FNR_g = FN_g / (TP_g + FN_g)
#   Gap = max_g(FNR_g) - min_g(FNR_g)
#
# Author: Teaching Project for Responsible AI
# ============================================================================

# ----------------------------------------------------------------------------
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# SECTION 2: CONFIGURATION AND CONSTANTS
# ----------------------------------------------------------------------------

# Define operational slices (groups to monitor)
OPERATIONAL_SLICES = ['Hospital_A', 'Hospital_B', 'Hospital_C', 'Device_Mobile', 'Device_Desktop']

# Alert configuration
ALERT_THRESHOLD = 0.10  # Alert if Gap > 10%
TIME_WINDOW_WEEKS = 4   # Rolling window of 4 weeks

# Seed for reproducibility
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# SECTION 3: DATA SIMULATION FUNCTIONS
# ----------------------------------------------------------------------------

def simulate_weekly_classification_data(
    slices: List[str],
    num_weeks: int = 8,
    base_fnr: float = 0.15,
    fnr_variation: float = 0.10,
    samples_per_slice: int = 500,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Simulate weekly classification results for multiple operational slices.
    
    Parameters:
    -----------
    slices : List[str]
        List of operational slice names (e.g., ['Hospital_A', 'Hospital_B'])
    
    num_weeks : int, default=8
        Number of weeks to simulate data for
    
    base_fnr : float, default=0.15
        Base false negative rate around which slices vary
    
    fnr_variation : float, default=0.10
        Maximum variation in FNR between slices (creates disparity)
    
    samples_per_slice : int, default=500
        Number of actual positive samples per slice per week
    
    seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: week, slice, TP, FN, FNR
    """
    np.random.seed(seed)
    
    data = []
    start_date = datetime(2026, 1, 1)
    
    for week_num in range(num_weeks):
        week_date = start_date + timedelta(weeks=week_num)
        
        for slice_name in slices:
            # Assign different FNR to each slice (simulating disparity)
            slice_index = slices.index(slice_name)
            slice_fnr = base_fnr + (slice_index / len(slices)) * fnr_variation
            
            # Add some random weekly variation
            weekly_fnr = slice_fnr + np.random.uniform(-0.02, 0.02)
            weekly_fnr = np.clip(weekly_fnr, 0.01, 0.50)
            
            # Calculate TP and FN based on FNR
            total_positives = samples_per_slice
            fn = int(weekly_fnr * total_positives)
            tp = total_positives - fn
            
            data.append({
                'week': week_num + 1,
                'week_date': week_date.strftime('%Y-%m-%d'),
                'slice': slice_name,
                'TP': tp,
                'FN': fn,
                'total_positives': total_positives
            })
    
    return pd.DataFrame(data)


# ----------------------------------------------------------------------------
# SECTION 4: FNR CALCULATION FUNCTIONS
# ----------------------------------------------------------------------------

def calculate_fnr_per_slice(tp: int, fn: int) -> float:
    """
    Calculate False Negative Rate for a single slice.
    Formula: FNR = FN / (TP + FN)
    """
    total = tp + fn
    if total == 0:
        return 0.0
    return fn / total


def calculate_fnr_for_all_slices(data: pd.DataFrame, week: int = None) -> Dict[str, float]:
    """Calculate FNR for all slices, optionally filtered by week."""
    if week is not None:
        filtered_data = data[data['week'] == week]
    else:
        filtered_data = data.groupby('slice').agg({'TP': 'sum', 'FN': 'sum'}).reset_index()
    
    fnr_dict = {}
    for _, row in filtered_data.iterrows():
        fnr = calculate_fnr_per_slice(row['TP'], row['FN'])
        fnr_dict[row['slice']] = fnr
    
    return fnr_dict


# ----------------------------------------------------------------------------
# SECTION 5: GAP METRIC CALCULATION
# ----------------------------------------------------------------------------

def calculate_fnr_gap(fnr_dict: Dict[str, float]) -> Tuple[float, str, str]:
    """
    Calculate the FNR Gap metric.
    Formula: Gap = max_g(FNR_g) - min_g(FNR_g)
    Returns: (gap_value, worst_slice_name, best_slice_name)
    """
    if not fnr_dict:
        return 0.0, None, None
    
    max_fnr = max(fnr_dict.values())
    min_fnr = min(fnr_dict.values())
    
    worst_slice = max(fnr_dict, key=fnr_dict.get)
    best_slice = min(fnr_dict, key=fnr_dict.get)
    
    gap = max_fnr - min_fnr
    
    return gap, worst_slice, best_slice


# ----------------------------------------------------------------------------
# SECTION 6: ROLLING WINDOW AGGREGATION
# ----------------------------------------------------------------------------

def calculate_rolling_window_fnr(
    data: pd.DataFrame,
    current_week: int,
    window_weeks: int = TIME_WINDOW_WEEKS
) -> Dict[str, float]:
    """Calculate FNR per slice using a rolling time window."""
    start_week = max(1, current_week - window_weeks + 1)
    end_week = current_week
    
    window_data = data[(data['week'] >= start_week) & (data['week'] <= end_week)]
    
    aggregated = window_data.groupby('slice').agg({'TP': 'sum', 'FN': 'sum'}).reset_index()
    
    fnr_dict = {}
    for _, row in aggregated.iterrows():
        fnr = calculate_fnr_per_slice(row['TP'], row['FN'])
        fnr_dict[row['slice']] = fnr
    
    return fnr_dict


# ----------------------------------------------------------------------------
# SECTION 7: ALERT THRESHOLD CHECKING
# ----------------------------------------------------------------------------

def check_alert_threshold(gap: float, threshold: float = ALERT_THRESHOLD) -> bool:
    """Check if the FNR gap exceeds the alert threshold."""
    return gap > threshold


def generate_alert_message(
    gap: float, worst_slice: str, best_slice: str,
    fnr_dict: Dict[str, float], week: int, threshold: float = ALERT_THRESHOLD
) -> str:
    """Generate a human-readable alert message."""
    alert_status = "🚨 ALERT FIRED" if gap > threshold else "✅ NO ALERT"
    
    message = f"""
{'='*60}
{alert_status}: FNR Gap Monitoring Report - Week {week}
{'='*60}

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     {gap:.4f} ({gap*100:.2f}%)
Threshold:     {threshold:.4f} ({threshold*100:.2f}%)
Status:        {"EXCEEDED" if gap > threshold else "WITHIN LIMITS"}

🏥 PER-SLICE FNR VALUES
-----------------------
"""
    
    sorted_slices = sorted(fnr_dict.items(), key=lambda x: x[1], reverse=True)
    
    for slice_name, fnr in sorted_slices:
        marker = "⚠️ WORST" if slice_name == worst_slice else ("✅ BEST" if slice_name == best_slice else "      ")
        message += f"  {marker} {slice_name}: {fnr:.4f} ({fnr*100:.2f}%)\n"
    
    message += f"""
📈 DISPARITY SUMMARY
--------------------
Worst Performing: {worst_slice} (FNR = {fnr_dict[worst_slice]:.4f})
Best Performing:  {best_slice} (FNR = {fnr_dict[best_slice]:.4f})
Disparity:        {gap:.4f} ({gap*100:.2f}%)

{'='*60}
"""
    
    return message


# ----------------------------------------------------------------------------
# SECTION 8: RUNBOOK IMPLEMENTATION
# ----------------------------------------------------------------------------

def execute_runbook(
    gap: float, worst_slice: str, best_slice: str,
    fnr_dict: Dict[str, float], week: int
) -> str:
    """Execute the runbook - immediate actions after alert fires."""
    runbook_log = f"""
{'#'*60}
# RUNBOOK EXECUTION - IMMEDIATE ACTIONS
{'#'*60}

📋 ALERT CONTEXT
----------------
Alert Fired:  Week {week}
Gap Value:    {gap:.4f} ({gap*100:.2f}%)
Worst Slice:  {worst_slice}
Best Slice:   {best_slice}

🔴 IMMEDIATE ACTIONS (Do within 24 hours)
-----------------------------------------

1. ⏹️  PAUSE DEPLOYMENT (if critical)
   - Consider pausing model for {worst_slice} if FNR > 25%
   - Current {worst_slice} FNR: {fnr_dict[worst_slice]:.4f}

2. 📧 NOTIFY STAKEHOLDERS
   - Send alert to ML team lead
   - Notify {worst_slice} operations manager
   - CC: Data Science team, Compliance team

3. 🔍 INITIAL INVESTIGATION
   - Check for data quality issues in {worst_slice}
   - Verify label quality for recent {worst_slice} data
   - Compare feature distributions: {worst_slice} vs {best_slice}

4. 📊 GATHER DIAGNOSTIC DATA
   - Export last 4 weeks of predictions for {worst_slice}
   - Pull confusion matrices per slice
   - Check for population shift

🟡 SHORT-TERM ACTIONS (Do within 1 week)
----------------------------------------

5. 🧪 ROOT CAUSE ANALYSIS
   - Perform error analysis on {worst_slice} false negatives
   - Check if model was trained with representative data
   - Investigate if operational changes occurred

6. 📝 DOCUMENT FINDINGS
   - Create incident report
   - Log in model monitoring dashboard
   - Update risk register

7. 🔧 MITIGATION OPTIONS
   - Consider slice-specific threshold adjustment
   - Evaluate need for model retraining
   - Plan A/B test for any changes

🟢 FOLLOW-UP ACTIONS (Do within 1 month)
----------------------------------------

8. 🔄 IMPLEMENT FIX
   - Apply approved mitigation
   - Monitor for improvement

9. ✅ VERIFY RESOLUTION
   - Confirm gap reduced below threshold
   - Document lessons learned

10. 📚 PROCESS IMPROVEMENT
    - Update monitoring thresholds if needed
    - Add new slices to monitoring if discovered

{'#'*60}
# END OF RUNBOOK
{'#'*60}
"""
    
    return runbook_log


# ----------------------------------------------------------------------------
# SECTION 9: MAIN MONITORING FUNCTION
# ----------------------------------------------------------------------------

def run_fnr_gap_monitoring(
    data: pd.DataFrame,
    threshold: float = ALERT_THRESHOLD,
    window_weeks: int = TIME_WINDOW_WEEKS,
    verbose: bool = True
) -> pd.DataFrame:
    """Run the complete FNR Gap monitoring pipeline."""
    max_week = data['week'].max()
    results = []
    
    print("\n" + "="*60)
    print("🔍 FNR GAP MONITORING SYSTEM - RUNNING")
    print("="*60)
    print(f"Threshold:     {threshold} ({threshold*100:.1f}%)")
    print(f"Time Window:   {window_weeks} weeks (rolling)")
    print(f"Slices:        {data['slice'].nunique()}")
    print(f"Weeks:         {max_week}")
    print("="*60 + "\n")
    
    for week in range(window_weeks, max_week + 1):
        fnr_dict = calculate_rolling_window_fnr(data, week, window_weeks)
        gap, worst_slice, best_slice = calculate_fnr_gap(fnr_dict)
        alert_fired = check_alert_threshold(gap, threshold)
        
        results.append({
            'week': week,
            'gap': gap,
            'worst_slice': worst_slice,
            'worst_fnr': fnr_dict.get(worst_slice, 0),
            'best_slice': best_slice,
            'best_fnr': fnr_dict.get(best_slice, 0),
            'alert_fired': alert_fired,
            'fnr_values': fnr_dict
        })
        
        if verbose:
            message = generate_alert_message(gap, worst_slice, best_slice, fnr_dict, week, threshold)
            print(message)
            
            if alert_fired:
                runbook = execute_runbook(gap, worst_slice, best_slice, fnr_dict, week)
                print(runbook)
    
    return pd.DataFrame(results)


# ----------------------------------------------------------------------------
# SECTION 10: MAIN EXECUTION
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    
    print("\n" + "#"*60)
    print("# FNR GAP MONITORING ALERT SYSTEM")
    print("# Detecting Unequal Harm Across Operational Slices")
    print("#"*60)
    
    # Step 1: Simulate weekly classification data
    print("\n📊 Step 1: Simulating weekly classification data...")
    data = simulate_weekly_classification_data(
        slices=OPERATIONAL_SLICES,
        num_weeks=8,
        base_fnr=0.12,
        fnr_variation=0.15,
        samples_per_slice=500
    )
    
    print("\n📋 Simulated Data Preview:")
    print(data.head(10).to_string(index=False))
    
    # Step 2: Run monitoring
    print("\n\n📈 Step 2: Running FNR Gap Monitoring...")
    results = run_fnr_gap_monitoring(
        data=data,
        threshold=ALERT_THRESHOLD,
        window_weeks=TIME_WINDOW_WEEKS,
        verbose=True
    )
    
    # Step 3: Summary
    print("\n" + "="*60)
    print("📊 MONITORING SUMMARY")
    print("="*60)
    
    alerts_fired = results['alert_fired'].sum()
    total_weeks = len(results)
    
    print(f"\nTotal Weeks Monitored: {total_weeks}")
    print(f"Alerts Fired:          {alerts_fired}")
    print(f"Alert Rate:            {alerts_fired/total_weeks*100:.1f}%")
    
    print("\n📈 Gap Values Over Time:")
    for _, row in results.iterrows():
        status = "🚨" if row['alert_fired'] else "✅"
        print(f"  Week {row['week']}: Gap = {row['gap']:.4f} {status}")
    
    print("\n" + "#"*60)
    print("# MONITORING COMPLETE")
    print("#"*60 + "\n")
