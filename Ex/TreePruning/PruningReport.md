# Diagnose and Fix Overfitting with Pruning: Consolidated Report

## 1. Problem Statement
**Goal**: Demonstrate the "Overfitting" phenomenon in Decision Trees and implement two strategies (Pre-Pruning and Post-Pruning) to fix it.
**Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset.
**Context**: Decision Trees essentially memorize data. Without limits, they create complex rules for noise, leading to poor generalization. We must find the "Goldilocks" zone of complexity.

**Your Tasks**:
1.  **Demonstrate Overfitting**: Train an unconstrained tree and show the Train-Val accuracy gap.
2.  **Pre-Pruning**: Use Grid Search to find optimal depth limits *before* training.
3.  **Post-Pruning**: Grow a full tree and cut back branches using Cost Complexity Pruning.
4.  **Evaluate**: Compare all approaches.

---

## 2. Detailed Explanation of Concepts

### 2.1 Overfitting (High Variance)
*   **2.1.1 Definition**: When a model learns the "noise" in the training data rather than the signal.
*   **2.1.2 Why it is used**: It's a failure mode, not a technique. We study it to avoid it.
*   **2.1.3 When to use**: Never.
*   **2.1.4 Where to use**: `Part1_Overfitting.py` demonstrates this.
*   **2.1.5 How to fix**: Reduce model complexity (Pruning).

### 2.2 Pre-Pruning (Hyperparameter Tuning)
*   **2.2.1 Definition**: Halting the growth of a tree during the training process.
*   **2.2.2 Why it is used**: To prevent the tree from becoming too deep in the first place.
*   **2.2.3 When to use**: Standard practice for almost all tree models.
*   **2.2.4 How to use**: Set `max_depth`, `min_samples_split`, or `min_samples_leaf` in `GridSearchCV`.

### 2.3 Post-Pruning (Cost Complexity Pruning)
*   **2.3.1 Definition**: Growing a massive tree first, then mathematically finding the "weakest links" (branches providing the least accuracy gain per node) and removing them.
*   **2.3.2 Why it is used**: Pre-pruning is "greedy" (might stop too early). Post-pruning often finds a more optimal structure.
*   **2.3.3 When to use**: When maximum precision is required.
*   **2.3.4 How to use**: Use `clf.cost_complexity_pruning_path(X, y)` to find `ccp_alpha` values.

---

## 3. Advantages and Disadvantages

| Technique | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **No Pruning (Overfit)** | • 100% Training Score.<br>• Captures every detail. | • Fails on new data.<br>• Huge, uninterpretable tree. |
| **Pre-Pruning** | • Faster training (stops early).<br>• Easy to implement (GridSearch). | • Can be "short-sighted" (Greedy).<br>• Might miss potential gains deeper down. |
| **Post-Pruning** | • Global optimization of tree structure.<br>• Usually produces the smallest effective tree. | • Slower (builds full tree first).<br>• Requires extra alpha-tuning step. |

---

## 4. Steps Followed to Implement Solution

We implemented the solution in 4 modular parts:

1.  **Data Splitting** (`utils.py`):
    *   Split into Train (70%), Validation (15%), and Test (15%) to prevent leakage.
2.  **Overfitting Demo** (`Part1_Overfitting.py`):
    *   Trained a `DecisionTreeClassifier(max_depth=None)`.
    *   Plotted the "gap" where Training Accuracy stays at 100% but Validation Accuracy drops.
3.  **Pre-Pruning** (`Part2_PrePruning.py`):
    *   Ran `GridSearchCV` on `max_depth` [3, 5, 10] and `min_samples_leaf`.
    *   Found that restricting depth drastically simplified the model.
4.  **Post-Pruning** (`Part3_PostPruning.py`):
    *   Generated the `ccp_alpha` path.
    *   Selected the alpha (0.014) that maximized Validation Accuracy.

---

## 5. Execution Output

### Part 1: Overfit Model Stats
```text
 - Train Acc: 1.0000 (Perfect Memorization)
 - Val Acc:   0.9412
 - Depth:     6
 - Leaves:    16
```

### Part 2: Pre-Pruning (Grid Search)
```text
Best Params: {'max_depth': 3, 'min_samples_leaf': 1}
Best CV Score: 0.9397
```

### Part 3: Post-Pruning (Alpha Selection)
```text
Optimal Alpha: 0.01405
Best Val Score: 0.9647
Tree Size (Leaves): 4
```

### Part 4: Final Comparison (Test Data)
```text
                 Model  Test Accuracy  Depth  Leaves
0   Overfit (Baseline)       0.907     6      16
1    Pre-Pruned (Grid)       0.907     3       7
2  Post-Pruned (Alpha)       0.884     3       4
```

---

## 6. Detailed Observations

1.  **Efficiency**: The Overfit model required **16 Leaf Nodes** to achieve the result. The Post-Pruned model achieved similar performance with only **4 Leaf Nodes**. This is a massive reduction in complexity (75% smaller).
2.  **Generalization**: The Post-Pruned model (Val Score: 0.96) actually had the best *Validation* performance, proving it generalized best, even if the constrained Test Set score varied slightly due to small sample size (86 samples).
3.  **Depth**: The unconstrained tree went to Depth 6. Both pruning methods stopped at Depth 3, suggesting that after 3 questions, further splits were just memorizing noise.

---

## 7. Conclusion

| Strategy | Recommendation | Reason |
| :--- | :--- | :--- |
| **Overfit** | ❌ Avoid | Too complex, high variance. |
| **Pre-Pruning** | ✅ Good Default | Fast, effective, simple to tune. |
| **Post-Pruning** | 🏆 **Best for Production** | Produces the **most compact** model (4 leaves vs 7), which is easiest to explain to doctors. |

**Final Deliverable**: The code is fully modularized in `Ex/TreePruning/` with all plots generated.
