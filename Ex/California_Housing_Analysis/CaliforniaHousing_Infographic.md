# 📊 Infographic: California Housing Transformation

## 🚨 The Critical Insight
> **5% of the data is a lie.**
> 965 houses are listed at **exactly $500,001**. This is a **Census Cap**, not a market price.

---

## 📉 The Data Funnel
*How we refined the raw input into a golden dataset.*

![Data Cleaning Funnel](data_cleaning_funnel.png)

*   **Raw Input**: 20,640 rows.
*   **IQR Filter**: Removes statistical anomalies.
*   **Clean Output**: 18,957 high-quality rows.

---

## ⚖️ Scale Comparison
*Why Standardization was necessary.*

![Scaling Comparison](scaling_comparison.png)

> **Takeaway**:
> *   **Left**: Population (up to 35,000) dwarfs Income (max 15).
> *   **Right**: After Scaling, both features are balanced (Z-Scores from -3 to +3).

---

## 📉 Visualizing Outliers (The Boxplot Method)
*How the Interquartile Range (IQR) identifies the "Extreme" values.*

![Outlier Detection Boxplot](outlier_boxplot.png)

*   **The Box**: Represents the "Normal" middle 50% of homes.
*   **The Whiskers**: Extending to reasonable limits (1.5x IQR).
*   **The Red Dots**: The outliers we removed (e.g., massive dormitories or administrative errors).

---

## 🔢 Key Statistics

| Metric | Before Cleaning | After Cleaning | Change |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 20,640 | 18,957 | 🔻 8.1% Dropped |
| **Max Income** | 15.0 | 15.0 | 0% |
| **Max Population** | **35,682** (City!) | **2,800** (Block) | ✅ Fixed Skew |
| **Price Cap** | Present | Present | ⚠️ Requires Label Removal |
