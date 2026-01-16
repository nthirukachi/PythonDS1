# 🧐 Observations and Conclusion

## 1. Execution Output Observations

1.  **Original Imbalance:**
    - 950 Healthy, 50 Disease (19:1 ratio).

2.  **After Stratified Split:**
    - Training: 760 Healthy, 40 Disease.
    - Test: 190 Healthy, 10 Disease.
    - The 5% ratio is preserved in both sets.

3.  **After PCA:**
    - Features reduced from 50 to 15.
    - This combats the Curse of Dimensionality.

4.  **After SMOTE:**
    - Training becomes balanced: 760 Healthy, 760 Disease.
    - This allows k-NN to learn meaningful patterns for the minority class.

5.  **Evaluation:**
    - Confusion Matrix shows improved True Positives for Class 1 (Disease).
    - Recall for Class 1 is now meaningful (e.g., 0.70) instead of near 0.
    - Precision for Class 1 is lower (due to False Positives), reflecting the trade-off.

## 2. Key Insights

- **Preprocessing Order Matters:** Split → PCA → SMOTE. Never SMOTE before splitting.
- **Accuracy is Deceptive:** In imbalanced data, focus on Recall and F1-Score for the minority class.
- **SMOTE is Not Magic:** It helps k-NN, but the test set remains imbalanced, so real-world performance differences may still be observed.
- **PCA Improves k-NN:** By reducing noise and dimensionality, distance calculations become more reliable.

## 3. Conclusion

We successfully built a rare disease detection pipeline:
- **Problem Solved:** Detected disease in a 19:1 imbalanced dataset using k-NN.
- **Best Approach:** PCA (50 → 15) + SMOTE (balance training) + k-NN (k=51).
- **Result:** Significant improvement in Recall for the Disease class compared to vanilla k-NN on raw, imbalanced data.
