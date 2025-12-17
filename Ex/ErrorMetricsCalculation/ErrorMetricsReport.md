# Error Metrics Analysis Report

## 1. Problem Statement
Using the housing prices dataset (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset), Explain Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error with examples from your housing prices prediction.

---

## 2. Detailed Explanation of Concepts

When we make predictions (e.g., predicting a house costs \$100k), we are almost always wrong by some amount (e.g., it actually costs \$105k). Error metrics help us quantify "how wrong" we are on average.

### Mean Absolute Error (MAE)
*   **Concept**: We take the difference between the Actual and Predicted value, remove the negative sign (Absolute), and take the average.
*   **Formula**: $\frac{1}{n}\sum |Actual - Predicted|$
*   **Why use it?**: It is **intuitive**. If MAE is \$5000, it means your prediction is wrong by \$5000 on average. It treats all errors equally.

### Mean Squared Error (MSE)
*   **Concept**: We take the difference, **square it**, and then take the average.
*   **Formula**: $\frac{1}{n}\sum (Actual - Predicted)^2$
*   **Why use it?**: Squaring makes big errors HUGE. If you confirm a \$10 error, it becomes 100. If you have a \$1000 error, it becomes 1,000,000.  This forces the model to fix outlier errors during training.
*   **Downside**: The unit is "Dollars Squared", which makes no sense to a human.

### Root Mean Squared Error (RMSE)
*   **Concept**: We just take the **Square Root** of the MSE.
*   **Why use it?**: It brings the unit back to "Dollars", so it's readable like MAE. However, because it came from squared errors, it still **penalizes large errors** more than MAE does. It is the industry standard.

---

## 3. Steps Followed to Implement

1.  **Library Setup**: Imported `pandas` for data and `sklearn.metrics` for the math.
2.  **Data Loading**: Loaded the housing data. Note that we had to use `../Housing.csv` because our script is in a subfolder.
3.  **Preprocessing**: Converted categories like 'furnished' to numbers so the regression math works.
4.  **Modeling**: Trained a Linear Regression model on 80% of the data.
5.  **Prediction**: Generated predictions for the remaining 20% (Test set).
6.  **Calculation**:
    *   **MAE**: Used `mean_absolute_error`.
    *   **MSE**: Used `mean_squared_error`.
    *   **RMSE**: Calculated `np.sqrt(MSE)`.

---

## 4. Observations of the Output

*(Hypothetical Output based on dummy data)*
*   **MAE**: ~1,200,000
*   **RMSE**: ~1,500,000

**Observation**:
You will notice that **RMSE is higher than MAE**.
*   **Why?**: This difference indicates that there are some predictions that are **very wrong** (outliers).
*   If every prediction was wrong by exactly \$1000, then MAE and RMSE would be identical.
*   Because we squared the errors to get RMSE, those few "very wrong" predictions drove the RMSE number up significantly.

---

## 5. Conclusion

*   **MAE** is the "purest" representation of average error. It answers "how close am I usually?"
*   **RMSE** is the "safety-first" representation. It answers "how bad is it when I'm wrong?"
*   For housing prices, **RMSE** is generally preferred because being off by a huge amount (e.g., underselling a mansion by \$5M) is much worse than being off by a small amount on many cheap houses. The squared term in RMSE captures this risk.
