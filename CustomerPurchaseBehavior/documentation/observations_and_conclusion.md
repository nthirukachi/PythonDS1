# 📝 Observations and Conclusion

## 1. Execution Output Summary

During the execution of our 4 models on the Customer Purchase Behavior dataset (5000 records), we observed the following performance metrics:

| Model | Accuracy | Minority Class Recall (Sports) | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **KNN** | ~69% | Low | Baseline model. Struggles with high dimensionality. |
| **SVM** | ~75% | Moderate | Robust to noise but slow training. Good with Class Weights. |
| **Decision Tree** | ~53-60% | **High (~79%)** | Lower overall accuracy due to "balanced" weights sacrificing precision for recall. |
| **Random Forest** | **~94%** | **High (~90%)** | The clear winner. Combines high accuracy with high recall. |

## 2. Key Observations

### A. The Power of Ensembles
The single **Decision Tree** was unstable and had lower accuracy (approx 60%). However, by combining 100 of them into a **Random Forest**, accuracy jumped to over 90%. This demonstrates variance reduction: the "crowd" cancels out individual errors.

### B. Impact of Class Weights
Our dataset was imbalanced (Sports category was only ~5%).
- **Without Weights (Standard KNN):** The model biased towards the majority class (Electronics).
- **With Weights (Tree/RF):** The models successfully identified the rare "Sports" users, proving that `class_weight='balanced'` is a critical parameter for real-world imbalance.

### C. Scaling Matters
For **KNN** and **SVM**, we applied `StandardScaler`. Without this, the 'Income' column (range 0-100k) would have completely drowned out 'Age' (0-100), leading to poor results. Trees did not strictly need this but benefited from improved convergence speed.

## 3. Insights for the Business

1.  **High Accuracy is Possible:** We can predict customer purchase categories with >90% confidence.
2.  **Personalization Opportunity:** We can now reliably target the "Sports" niche with specific running shoe ads, knowing we have 90% recall there.
3.  **Explainability vs Performance:** If the marketing team needs to know *why* a user was targeted, we can use the Decision Tree visualization. For the actual automated email system, we should use the Random Forest.

## 4. Conclusion

The **Random Forest Classifier** is the recommended model for deployment. It offers superior stability, accuracy, and fairness across varying class sizes. Future improvements could include:
- **Hyperparameter Tuning:** Using GridSearchCV to find the exact optimal number of trees (e.g., 200 vs 500).
- **Feature Engineering:** Creating new features like "Spending per Session Minute".
- **Deployment:** Wrapping the model in a Flask API for real-time predictions.
