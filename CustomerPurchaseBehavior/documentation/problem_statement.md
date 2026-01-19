# 🧩 Problem Statement: Customer Purchase Behavior Analysis

## 1. What problem is being solved?
E-commerce companies collect vast amounts of data on their customers (Age, Income, Browsing history). However, raw data doesn't tell us *what* they will buy next. The problem is to **automatically classify** a customer into one of 5 purchase categories:
- **0: Electronics** (Dominant category, ~45% of users)
- **1: Fashion**
- **2: Home**
- **3: Books**
- **4: Sports** (Rare category, ~5% of users)

Solving this allows the company to show personalized recommendations (e.g., showing running shoes to a predicted "Sports" buyer), which increases sales and customer satisfaction.

## 2. Steps to Solve the Problem
We follow a standard Data Science Pipeline:

1.  **Data Loading & Exploration (EDA):**
    -   We load 5000 records.
    -   We discover missing values in Income/Spending and a severe class imbalance.
2.  **Preprocessing:**
    -   **Imputation:** Filling missing values with the average.
    -   **Encoding:** Converting "Mobile"/"Desktop" to 0/1.
    -   **Scaling:** Adjusting Income ($50k) and Age (30) to be on the same scale (0-1 range).
3.  **Model Selection & Training:**
    -   **KNN:** Finds similar customers.
    -   **SVM:** Draws mathematical boundaries.
    -   **Decision Tree:** Creates a rule-based flowchart.
    -   **Random Forest:** Combines many trees for robust prediction.
4.  **Evaluation:**
    -   We check Accuracy (Overall correctness).
    -   We check Recall (Ability to find rare classes like Sports).

## 3. Expected Output
- **Python Scripts:** 5 standalone scripts for each model and a comparison.
- **Jupyter Notebooks:** Detailed teaching guides for each model.
- **Visualizations:** Confusion Matrices, Tree Charts, and Comparison Plots.
- **Slide Decks:** 5 PDF presentations summarizing the logic.
- **Final Verdict:** Recommendation of the best model (Random Forest).
