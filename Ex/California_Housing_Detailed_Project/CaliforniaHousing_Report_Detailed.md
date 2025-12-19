
# California Housing Analysis: Technical Report

## 1. Problem Statement
The goal of this project is to prepare the California Housing dataset for machine learning. This involves a rigorous process of **EDA (Exploratory Data Analysis)** to understand the data, **Outlier Detection** to identify anomalies, **Data Cleaning** to remove those anomalies, and **Feature Scaling** to normalize the data range. This ensures that downstream ML algorithms (like Linear Regression or Neural Networks) perform optimally.

## 2. Concept Explanations (Strict Schema)

### 2.1 Pandas Library
*   **Definition**: An open-source Python library providing high-performance, easy-to-use data structures and data analysis tools.
*   **Why it is used**: It is the industry standard for tabular data manipulation. It handles loading, filtering, and transforming large datasets efficiently.
*   **When to use**: Whenever working with structured data (CSV, SQL, Excel).
*   **Where to use**: Throughout the entire script (loading data, calculating stats).
*   **How to use**: `import pandas as pd`, then use `pd.DataFrame()`.
*   **How it works**: Uses NumPy under the hood for speed but provides labeled axes (columns/index) for usability.
*   **Visual Summary**:
    ![Data Funnel](cal_housing_data_funnel.png)

### 2.2 IQR (Interquartile Range) Method
*   **Definition**: A statistical measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles (Q3 - Q1).
*   **Why it is used**: It is robust against extreme outliers compared to standard deviation.
*   **When to use**: When data is skewed or not normally distributed.
*   **Where to use**: `detect_outliers_iqr` function.
*   **How to use**: Calculate `Q3 - Q1`, then define fences at `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR`.
*   **How it works**: Any data point outside the fences is flagged as an outlier.
*   **Visual Summary**:
    ![IQR Boxplot Schematic](cal_housing_boxplot_schematic.png)

### 2.3 Z-Score Outlier Detection
*   **Definition**: A measure of how many standard deviations below or above the population mean a raw score is.
*   **Why it is used**: To identify outliers in Normally Distributed data.
*   **When to use**: When the data follows a Bell Curve.
*   **Where to use**: `detect_outliers_zscore`.
*   **How to use**: `z = (x - mean) / std`. Threshold usually > 3.
*   **How it works**: Standardizes the data; values far from 0 are outliers.
*   **Visual Summary**:
    ```text
      -3s   -2s   -1s   Mean   +1s   +2s   +3s
       |     |     |      |      |     |     |
    [Reject] [           Accept            ] [Reject]
    ```

### 2.4 Feature Scaling (StandardScaler)
*   **Definition**: A preprocessing technique that normalizes the range of independent variables or features of data.
*   **Why it is used**: Many ML algorithms (Gradient Descent, KNN) compute distances/gradients that are sensitive to scale.
*   **When to use**: Before training the model.
*   **Where to use**: Final step of the script.
*   **How to use**: `scaler.fit_transform(data)`.
*   **How it works**: Centers distribution around 0 with a unit variance.
*   **Visual Summary**:
    ![Feature Scaling Viz](cal_housing_scaling_viz.png)

## 3. Advantages & Disadvantages of Methods

| Method | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **IQR (Outliers)** | Robust to extreme deviations; Doesn't assume Normal distribution. | May capture "valid" extreme values in heavy-tailed distributions. |
| **Z-Score (Outliers)** | Simple mathematically; Good for Gaussian data. | Sensitive to the mean/std (which outliers themselves affect). |
| **Standard Scaling** | Preserves outliers/shape; Optimal for most ML algos. | Doesn't bound data to fixed range (unlike MinMax). |

## 4. Implementation Steps
1.  **Setup**: Import libraries (`sklearn`, `pandas`, `numpy`).
2.  **Load**: Fetch `california_housing` data.
3.  **Explore**: Use `head()`, `describe()`, `info()` to inspect structure.
4.  **Detect**:
    *   Percentiles for extreme values.
    *   IQR for statistical spread.
    *   Logic for semantic checks (Bedrooms > Rooms).
5.  **Clean**: Remove outliers from `AveRooms` and `Population`.
6.  **Scale**: Apply `StandardScaler` to numerical columns.
7.  **Verify**: Check that mean is ~0 and std is ~1.

## 5. Execution Output (Expected)
*   **Dataset Shape**: `(20640, 9)`
*   **Outliers Detected**:
    *   Method 1 (Percentiles): Shows extremes (e.g. 5.0000 capped).
    *   Method 2 (IQR): ~1000+ outliers typically found in price/rooms.
    *   Method 5 (Logical): Small number of erroneous entries.
*   **Cleaning**: Shape reduces from 20640 to ~19,000 range.
*   **Scaling**:
    *   Before: MedInc Mean ~3.8, Std ~1.9.
    *   After: MedInc Mean ~0.0, Std ~1.0.

## 6. Detailed Observations
*   **Capped Values**: The target `MedHouseValue` is clearly capped at 5.00001 (representing $500k+). This is a known quirk of this dataset and introduces a "wall" in the distribution.
*   **Skewed Features**: `AveRooms` and `Population` are heavily right-skewed. The IQR method heavily prunes these, which is good for linear models but might remove valid "mansion" or "dense city block" data.
*   **Scaling Impact**: The transformation successfully zero-centered the data. This is critical because `Population` (range 1-35000) would otherwise dominate `AveRooms` (range 1-140) in any distance-based calculation.

## 7. Conclusion
The analysis successfully sanitized the California Housing dataset. By removing statistical anomalies and standardizing the feature space, we have created a high-quality input dataset that will allow Machine Learning models to converge faster and generalize better.
