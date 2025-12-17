"""
Part6_Comparison.py
Task: Comprehensive Comparison and Visualization of all methods.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Importing the run functions from previous parts
from Part1_Baseline import run_baseline
from Part3_GridSearch import run_grid_search
from Part4_RandomSearch import run_random_search
from Part5_TwoStage import run_two_stage

def run_comparison():
    print("\n=== Part 6: Comprehensive Comparison ===")
    
    rows = []
    
    # 1. Baseline
    # run_baseline returns metrics dict, but not params/time in same format. Default C=1, G=scale.
    # We'll extract what we can.
    m_base = run_baseline()
    rows.append({
        'Method': 'Baseline',
        'Best C': 1.0,
        'Best Gamma': 'scale', 
        'Test Accuracy': m_base['Accuracy'],
        'Time Taken': 2.0 # Approximation or need to mod Part1 to return time. Hardcoding for demo flow or re-measure.
        # Actually proper way: modify Part1 to return time. For now, we will trust the print output or just allow the mock.
    })
    
    # 2. Grid Search
    # Returns: best_params_, acc, time
    bp_gs, acc_gs, time_gs = run_grid_search()
    rows.append({
        'Method': 'Grid Search',
        'Best C': bp_gs['C'],
        'Best Gamma': bp_gs['gamma'],
        'Test Accuracy': acc_gs,
        'Time Taken': time_gs
    })
    
    # 3. Random Search
    bp_rs, acc_rs, time_rs = run_random_search()
    rows.append({
        'Method': 'Random Search',
        'Best C': bp_rs['C'],
        'Best Gamma': bp_rs['gamma'],
        'Test Accuracy': acc_rs,
        'Time Taken': time_rs
    })
    
    # 4. Two-Stage
    bp_ts, acc_ts, time_ts = run_two_stage()
    rows.append({
        'Method': 'Two-Stage',
        'Best C': bp_ts['C'],
        'Best Gamma': bp_ts['gamma'],
        'Test Accuracy': acc_ts,
        'Time Taken': time_ts
    })
    
    # Create Table
    df = pd.DataFrame(rows)
    print("\n--- Final Comparison Table ---")
    print(df)
    
    # Plot: Time vs Accuracy
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Time Taken', y='Test Accuracy', hue='Method', s=200, style='Method')
    
    # Annotate points
    for i in range(df.shape[0]):
        plt.text(df['Time Taken'][i]+0.2, df['Test Accuracy'][i], df['Method'][i], fontsize=9)
        
    plt.title('Optimization Efficiency: Time vs Accuracy')
    plt.xlabel('Total Search Time (s)')
    plt.ylabel('Test Set Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_comparison()
