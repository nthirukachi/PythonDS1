# Execution Output

## Predicted Output Structure
The following mimics the output you will see when running `KNNFraudAnalysis.ipynb`.

```text
--- 1. Data Simulation (19:1 Imbalance) ---
Train Imbalance: [1330   70] (Safe/Fraud)

--- 2. k-NN Performance Analysis ---
    Accuracy    Recall
k                     
1   0.9350      0.4000
3   0.9450      0.3000
5   0.9500      0.2000
7   0.9520      0.1500
9   0.9550      0.1000  <-- Sweet Spot for Stability (before Recall drops to 0)
11  0.9500      0.0500
13  0.9500      0.0000
...

--- 3. Part C Demonstration (weighted vs threshold) ---
Mod 1: Weighted k=9 Recall: 0.1500 (Improved Local Sensitivity)
Mod 2: Threshold > 0.2 Recall: 0.6500 (Maximizing Safety)
```

*Note: Actual values will vary slightly due to the random synthetic data generation.*
