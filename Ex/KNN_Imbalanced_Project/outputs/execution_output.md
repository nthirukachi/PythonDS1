# Execution Output

## Predicted Output Structure
The following mimics the output you will see when running `KNN_Imbalanced_Solution.ipynb`.

```text
Original Class Distribution: [950  50]

Training set shape: (800, 50)
Test set shape: (200, 50)
Train Class Distribution: [760  40]

X_train_pca shape: (800, 15)
X_test_pca shape: (200, 15)

Resampled Class Distribution: [760 760]

k-NN trained with k=51

--- Model Evaluation ---
Confusion Matrix:
[[150  40]
 [  3   7]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.98      0.79      0.87       190
         1.0       0.15      0.70      0.25        10

    accuracy                           0.79       200
   macro avg       0.56      0.74      0.56       200
weighted avg       0.94      0.79      0.84       200
```

*Note: Actual values will vary slightly due to the random nature of synthetic data generation. The key observation is that Class 1 (Disease) Recall is now meaningful (e.g., 0.70) instead of 0.*
