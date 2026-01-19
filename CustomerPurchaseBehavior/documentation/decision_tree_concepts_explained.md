# 📘 Decision Tree Concepts Explained

## 1. Decision Tree (CART)

### Definition
A **Decision Tree** is a supervised learning method that predicts the value of a target variable by learning simple decision rules inferred from the data features. It looks like a flowchart where each internal node represents a "test" on an attribute (e.g., whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label.

### Why / When / Where?
-   **Why:** It mirrors human decision-making. It is "White Box" AI—you can see exactly why it made a decision.
-   **When:** When explainability is the #1 priority (e.g., Why was my loan rejected?).
-   **Where:** Medical Diagnosis (Symptom -> Disease), Credit Scoring, Customer Segmentation.

### How to use it?
1.  **Ask Questions:** The algorithm searches for the best "Yes/No" question to split the data (e.g., "Is Income > $50k?").
2.  **Split:** Divide data into two piles based on the answer.
3.  **Repeat:** Ask a new question for each pile.
4.  **Stop:** When the pile is pure (only one class left) or we reach a limit (max depth).

```python
from sklearn.tree import DecisionTreeClassifier
# max_depth=4 prevents the tree from becoming a giant monster
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)
```

### How it works internally
1.  **Gini Impurity / Entropy:** To pick the "Best Question", it calculates how "mixed" a pile is.
    -   A pile with 50 apples and 50 oranges is **Impure** (High Gini).
    -   A pile with 100 apples is **Pure** (Gini = 0).
2.  **Greedy Search:** It tries every possible split on every column and picks the one that reduces Impurity the most (Information Gain).

### Visual Summary
Imagine playing **"20 Questions"**.
-   **Q1:** Is it an animal? (Yes)
-   **Q2:** Does it bark? (Yes)
-   **Prediction:** Dog.
-   The Decision Tree creates this exact map automatically from data.

### Advantages
-   **Interpretability:** Can be visualized as a chart.
-   **No Scaling Needed:** It doesn't care about the magnitude of numbers (unlike SVM/KNN).
-   **Handle Mixed Data:** Works well with both numbers and categories.

### Disadvantages
-   **Overfitting:** Single trees tend to grow overly complex branches that memorize noise. They don't generalize well.
-   **Instability:** Changing one row of data can alter the entire tree structure.

---

## 2. Class Weights ('balanced')

### Definition
**Class Weighting** is a technique to handle imbalanced datasets (where one outcome is rare, like fraud or purchase).

### Why / When / Where?
-   **Why:** Standard algorithms care about Accuracy. If 99% of transactions are safe, predicting "Safe" 100% of the time gives 99% accuracy. But we miss the fraud!
-   **When:** When the target classes are unequal (e.g., 5000 users, but only 100 buy).

### How to use it?
```python
model = DecisionTreeClassifier(class_weight='balanced')
```

### How it works internally
It modifies the Gini Impurity calculation. It effectively says: "Making a mistake on the Minority Class costs 100 points. Making a mistake on the Majority Class costs 1 point." This forces the model to pay attention to the little guy.

### Advantages
-   Fixes bias without needing to generate fake data.

### Disadvantages
-   May increase False Positives (predicting fraud when it's safe) because the model becomes paranoid.
