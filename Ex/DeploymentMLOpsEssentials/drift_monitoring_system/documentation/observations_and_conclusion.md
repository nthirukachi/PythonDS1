# Observations and Conclusion

## 1. Execution Output Summary
We ran the monitoring pipeline on 3 different batches of data:

-   **Baseline Batch**: Represented normal behavior.
    -   *Result*: **PASS**. Both Data Quality and Drift Checks passed. System Healthy.
-   **Batch 1 (Data Drift)**: Represented a "High Inflation" scenario (Income increased).
    -   *Result*: **FAIL (Alert Triggered)**.
    -   *Detection*: The **KS-Test** correctly identified that the distribution of "Income" changed significantly (p-value < 0.05).
    -   *Action*: The system suggested verifying data integrity or retraining.
-   **Batch 2 (Concept Drift)**: Represented a "Policy Change" (Stricter approval rules), but Income remained normal.
    -   *Result*: **PASS (False Negative!)**.
    -   *Observation*: The input data distribution ($P(X)$) looked exactly like the baseline. The monitoring system said "System Healthy".
    -   *Reality*: The model was predicting approvals based on OLD rules, which is wrong.

## 2. Key Observations
1.  **Drift Checks are not Magic**: Checking $P(X)$ (Input Drift) is easy and cheap. It catches data issues like broken sensors or major population shifts.
2.  **The "Silent Killer"**: Concept Drift ($P(Y|X)$) is invisible to input-only monitoring. If the world's logic changes but the data looks the same, your Drift Monitor will stay green while your business loses money.
3.  **Alerting is Critical**: A p-value means nothing to a business stakeholder. Translating "p < 0.05" to "Distribution Shift Detected: Verify Data" is the bridge between Math and Engineering.

## 3. Conclusion
We successfully designed a "Security Guard" for our ML model. It is perfect for detecting **Environment Changes** (Data Drift) but has a blind spot for **Logic Changes** (Concept Drift).

**Recommendation**: To fix the blind spot, we must add **Performance Monitoring**. We need to collect feedback (did the loan default?) and check the Accuracy/F1-Score over time.
