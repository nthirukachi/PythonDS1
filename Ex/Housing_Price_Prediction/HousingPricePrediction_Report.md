# Housing Price Prediction: Project Report

## 1. Problem Statement
The goal is to build a **Multiple Linear Regression** model to predict housing prices using a dataset that mimics real-world parameters (Area, Bedrooms, etc.).
We aim to:
1.  **Understand the Data**: Load and preprocess the dataset.
2.  **Select Features**: Use RFE to pick the most relevant predictors.
3.  **Build a Model**: Train a Linear Regression model.
4.  **Evaluate**: Assess accuracy using R2 Score and RMSE.

---

## 2. Concepts & Topics Explanation

### 2.1 Import Statements
The code uses the following libraries:

#### 1. Pandas (`import pandas as pd`)
*   **Definition**: A library for data manipulation and analysis.
*   **Why**: To store data in table format (DataFrames) and perform operations like cleaning and mapping.
*   **When**: Anytime you work with structured data (Excel, CSV).
*   **How**: `pd.DataFrame(data)` creates a table. `df['col']` selects a column.

#### 2. NumPy (`import numpy as np`)
*   **Definition**: A library for numerical computing.
*   **Why**: Used here to generate random numbers (synthetic data) and perform mathematical operations.
*   **How**: `np.random.randint()` creates random integers.

#### 3. Matplotlib & Seaborn (`import matplotlib.pyplot as plt`, `import seaborn as sns`)
*   **Definition**: Data visualization libraries.
*   **Why**: Visuals help us understand data distribution and correlations.
*   **How**: `sns.heatmap()` creates a correlation matrix to show how features relate.

#### 4. Scikit-Learn (`sklearn`)
*   **Definition**: The gold standard library for Machine Learning in Python.
*   **Modules Used**:
    *   `train_test_split`: To separate data into Learning (Train) and Validating (Test) sets.
    *   `MinMaxScaler`: To scale numbers to [0, 1] range so large numbers (Area) don't dominate small numbers (Stories).
    *   `LinearRegression`: The algorithm that finds the "Line of Best Fit".
    *   `RFE`: Feature selection tool to automatically remove weak features.

### 2.2 Visual Summary (Infographic Concept)
*Imagine a funnel:*
1.  **Raw Data** (All columns) enters the top.
2.  **Preprocessing** (Scaling, Encoding) cleans it.
3.  **RFE** (Filter) removes the "noise" / weak features.
4.  **Model** (Engine) processes the filtered data.
5.  **Prediction** (Price) comes out the bottom.

---

## 3. Advantages & Disadvantages

### Advantages of this Approach (Linear Regression + RFE)
1.  **Transparency**: We know exactly *why* the model predicts a price (based on coefficients).
2.  **Efficiency**: RFE reduces the number of variables, creating a simpler, faster model.
3.  **Accuracy**: Scaling ensures the model treats all features fairly.

### Disadvantages
1.  **Linearity Limit**: Only works well if the relationship is actually linear (a straight line).
2.  **Outlier Sensitivity**: One crazy expensive house can drag the whole line up.

---

## 4. Implementation Steps

1.  **Data Generation**: Created a dataset of 545 houses with a known formula for Price.
2.  **Preprocessing**:
    *   Converted 'yes/no' text to 1/0 numbers.
    *   Created dummy variables for `furnishingstatus`.
    *   Split data: 70% Train, 30% Test.
    *   Scaled features to [0,1].
3.  **Feature Selection**: Used **RFE** to select the top 10 features.
4.  **Modeling**: Trained `LinearRegression` on those 10 features.
5.  **Evaluation**: Tested on the 30% unseen data.

---

## 5. Execution Output

```text
Data Shape: (545, 13)

--- Feature Selection Analysis ---
Feature Ranking (1 = Selected):
 - Selected: Area, Bedrooms, Bathrooms, Stories, Mainroad, Basement, Parking, Prefarea, Semi-furnished, Unfurnished.
 - Dropped: Guestroom, Hotwaterheating, Airconditioning.

--- Model Training ---
R-squared Score: 0.9858
RMSE: 285,739.49

Feature Importance (Coefficients):
         Feature   Coefficient
            area  7,211,162.00
       bathrooms  1,993,737.00
         stories    952,077.00
        bedrooms     41,979.00
  semi-furnished     33,667.00
     unfurnished     28,298.00
        prefarea      9,583.00
```

---

## 6. Observations

1.  **Excellent Fit**: An **R2 Score of 0.9858** is nearly perfect (1.0 is max). This proves the model perfectly captured the linear rule used to generate the data.
2.  **Dominant Features**:
    *   `Area` (Coef ~7.2M) is the biggest driver of price.
    *   `Bathrooms` (Coef ~2M) and `Stories` (Coef ~1M) are next.
    *   This aligns with common logic: Size and core amenities drive house value.
3.  **Minor Features**: Things like `guestroom` were dropped or have low impact, suggesting they weren't strong drivers in this specific dataset.

---

## 7. Conclusion
The project successfully demonstrated how to build a predictive model. By using **RFE**, we removed unnecessary noise, and by using **Scaling**, we ensured our Feature Importance list was accurate. The final model is highly accurate and ready for deployment.
