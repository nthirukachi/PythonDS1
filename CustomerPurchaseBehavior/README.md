# 🛍️ Customer Purchase Behavior Analysis

## Project Overview
This project applies Machine Learning to predict customer purchase categories (Electronics, Fashion, Home, Books, Sports) based on demographic and behavioral data. Ideally suited for students and beginners, this repository contains step-by-step implementations of 4 major algorithms.

## 📂 Structure
```
CustomerPurchaseBehavior/
├── data/                   # Dataset
├── notebook/               # Teaching Notebooks (Detailed Explanations)
├── documentation/          # Problem Statement & Concepts
├── slides/                 # PDF Presentations & Markdown Source
├── src/                    # Python Scripts (Modular)
└── outputs/                # Generated Plots & Images
```

## 🚀 How to Run

### 1. Environment Setup
Ensure you are using the project's virtual environment:
```powershell
& c:/nagpython/demouv/.venv/Scripts/Activate.ps1
```

### 2. Run Individual Models
Each model is self-contained. Run them to see the output and generate plots.
```powershell
# KNN
python src/knn_model.py

# SVM
python src/svm_model.py

# Decision Tree
python src/decision_tree_model.py

# Random Forest
python src/random_forest_model.py
```

### 3. Compare Models
To see a head-to-head showdown:
```powershell
python src/combined_models.py
```

## 📊 Key Results
- **Champion Model:** Random Forest (~94% Accuracy).
- **Key Insight:** Handling class imbalance with `class_weight='balanced'` was crucial for detecting rare customer segments.

## 📚 Educational Resources
- Check `notebook/` for line-by-line code explanations.
- Check `slides/` for visual summaries of each algorithm.
- Check `documentation/` for deep dives into concepts like "Imputation" and "Hyperplanes".

---
*Created by AI Agent (Antigravity)*
