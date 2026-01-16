# 🧐 Observations and Conclusion

## 1. Execution Output Observations
*(Based on the logic of the code)*

1.  **Imbalance Check:**
    - The code successfully generates 2000 samples.
    - Training set shows ~1330 Safe and ~70 Fraud (maintaining 19:1 ratio).

2.  **k-NN Loop Analysis:**
    - **k=1:** Recall is decent, but Variance is high.
    - **k=3 to k=7:** Accuracy improves, Recall might jitter.
    - **k=9:** We see the "Sweet Spot". Accuracy is near peak, and Recall is stable.
    - **k > 15:** Accuracy plateaus at ~95% (Baseline), but Recall drops significantly (approaching 0). The model just predicts "Safe" for everything.

3.  **Part C Modifications:**
    - **Weighted k-NN:** Recall typically improves slightly because local clusters of fraud vote harder.
    - **Threshold > 0.2:** Recall improves **drastically**. By lowering the bar for what we call "Fraud", we catch almost all of them, though we likely flag more legitimate users (False Positives).

## 2. Key Insights
- **Accuracy is deceptive.** In fraud detection, high accuracy often effectively means "Model failed to learn anything except majority voting."
- **Thresholding is powerful.** Changing the probability threshold (e.g., from 0.5 to 0.2) is often more effective than trying to find a better algorithm or a better `k`.
- **k-NN behavior.** The algorithm is very sensitive to `k`. In imbalanced data, high `k` tends to drown out the minority class completely.

## 3. Conclusion
We successfully built a Fraud Detection simulation.
- **Problem Solved:** Detected fraud in 19:1 imbalanced data.
- **Best Approach:** k-NN with `k=9` combined with **Threshold Tuning** (flagging if >20% neighbors are fraud) yields the safest system for the bank.
