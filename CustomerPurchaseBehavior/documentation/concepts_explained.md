# 📚 Concepts Explained

## 1. Data Preprocessing

### Imputation (Handling Missing Data)
- **Definition:** The process of replacing missing data with substituted values.
- **Why:** Machine Learning algorithms generally cannot handle blank spaces (NaN).
- **How used:** We used `SimpleImputer(strategy='mean')` to replace missing Income with the average Income.

### One-Hot Encoding
- **Definition:** Converting categorical variables (text) into a binary matrix (0s and 1s).
- **Why:** Math equations work on numbers, not words like "Gold Member".
- **Example:** "Device" column becomes "Device_Mobile", "Device_Desktop".

### Feature Scaling (StandardScaler)
- **Definition:** Transforming data so that it has valid range (mean=0, variance=1).
- **Why:** Crucial for distance-based algorithms like KNN and SVM. Without it, "Income" (Range 0-100,000) would dominate "Age" (Range 0-100).
- **Analogy:** Comparing apples to apples, not apples to skyscrapers.

---

## 2. Machine Learning Algorithms

### K-Nearest Neighbors (KNN)
- **Concept:** "Birds of a feather flock together."
- **How it works:** To predict a new user's category, it looks at the 'k' closest existing users. Majority wins.
- **Pros:** Simple. **Cons:** Slow with big data.

### Support Vector Machine (SVM)
- **Concept:** "The Widest Street."
- **How it works:** Tries to find a line (or hyperplane) that separates classes with the maximum margin/gap.
- **Kernel Trick:** Projects data into higher dimensions to separate complex overlaps.

### Decision Tree
- **Concept:** "20 Questions."
- **How it works:** Splits data based on rules (e.g., "Is Spending > 500?").
- **Pros:** Interpretable (White Box). **Cons:** Prone to memorizing data (Overfitting).

### Random Forest
- **Concept:** "Wisdom of Crowds."
- **How it works:** Builds 100 random Decision Trees. If 70 trees say "Sports", the final prediction is "Sports".
- **Pros:** Very accurate and robust. **Cons:** Hard to interpret.

---

## 3. Evaluation Metrics

### Accuracy
- **Definition:** % of correct predictions.
- **Warning:** Misleading in imbalanced data. If 95% of users are Class 0, a model that guesses "Class 0" for everyone has 95% accuracy but is useless.

### Precision & Recall
- **Precision:** "When it predicts Sports, is it actually Sports?" (Quality).
- **Recall:** "Of all actual Sports fans, how many did we find?" (Quantity).

### Confusion Matrix
- **Definition:** A table showing True vs Predicted labels.
- **Usage:** Helps pinpoint specific confusions (e.g., "Did we confuse Home with Electronics?").
