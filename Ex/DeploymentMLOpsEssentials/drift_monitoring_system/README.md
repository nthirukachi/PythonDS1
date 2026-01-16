# Drift Monitoring system

**Project**: Design a monitoring plan for a production ML system.

## 📂 Project Structure

```
c:\nagpython\demouv\drift_monitoring_system\
├── 📁 notebook/                # Jupyter Notebooks (Teaching Mode)
│   └── drift_monitoring_system.ipynb
│
├── 📁 documentation/           # Detailed Explanations
│   ├── problem_statement.md
│   ├── concepts_explained.md
│   └── observations_and_conclusion.md
│
├── 📁 slides/                  # Presentation
│   ├── notebooklm_style_slides.md
│   └── notebooklm_style_slides.pdf
│
├── 📁 src/                     # Source Code
│   └── drift_demo.py
│
└── 📁 outputs/                 # Execution Results
```

## 🚀 How to Run

### 1. Python Script
Run the source code to see the raw output of the monitoring system.
```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe src/drift_demo.py
```

### 2. Jupyter Notebook
Open VS Code or Jupyter Lab to study the step-by-step teaching notebook.
```powershell
# Open VS Code
code notebook/drift_monitoring_system.ipynb
```

## 🧠 Key Concepts Covered
- **Data Drift ($P(X)$)**: Detected via **KS Test** and **Mean Shift**.
- **Concept Drift ($P(Y|X)$)**: Logic changes that are HARD to detect without ground truth.
- **Data Quality**: Null checks and Range checks.
- **Alerting**: Automated plain-language alerts.
