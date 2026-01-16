# Canary Rollout Prediction Shift Analysis

## 📌 Project Overview

This is a **comprehensive teaching project** that demonstrates how to detect, analyze, and respond to prediction distribution shifts during a canary deployment of a machine learning classifier model.

---

## 🎯 Learning Objectives

After studying this project, you will understand:

1. **Canary Deployment Strategy** - How ML models are gradually rolled out to production
2. **Prediction Drift Detection** - How to detect when model outputs change unexpectedly
3. **Covariate Shift** - What happens when input data distribution changes
4. **Population Stability Index (PSI)** - How to measure distribution drift
5. **Kolmogorov-Smirnov Test** - Statistical test for comparing distributions
6. **Rollback Decisions** - When to continue, pause, or rollback a deployment

---

## 📁 Folder Structure

```
📁 canary_rollout_prediction_shift/
│
├── 📁 notebook/
│   └── canary_rollout_prediction_shift.ipynb    # Teaching notebook
│
├── 📁 documentation/
│   ├── problem_statement.md                      # Problem definition
│   ├── concepts_explained.md                     # Detailed concept explanations
│   └── observations_and_conclusion.md            # Analysis results
│
├── 📁 slides/
│   ├── notebooklm_style_slides.md               # Slide deck (Markdown)
│   └── notebooklm_style_slides.pdf              # Slide deck (PDF)
│
├── 📁 src/
│   └── canary_rollout_demo.py                   # Source Python script
│
├── 📁 outputs/
│   ├── execution_output.md                       # Script execution results
│   └── sample_outputs/                           # Sample output files
│
└── README.md                                     # This file
```

---

## 🚀 How to Run

### Option 1: Run Python Script

```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe c:/nagpython/demouv/canary_rollout_prediction_shift/src/canary_rollout_demo.py
```

### Option 2: Open Jupyter Notebook

```powershell
# Activate virtual environment
& c:/nagpython/demouv/.venv/Scripts/Activate.ps1

# Start Jupyter
jupyter notebook c:/nagpython/demouv/canary_rollout_prediction_shift/notebook/canary_rollout_prediction_shift.ipynb
```

---

## 📖 Scenario Description

### The Problem

A machine learning team deploys a new classifier model using a **canary rollout strategy**:
- 10% of traffic is routed to the new model
- 90% of traffic remains on the stable baseline model

### Observations After 2 Hours

| Metric | Status |
|--------|--------|
| **Latency** | Normal ✅ |
| **Error Rate** | Normal ✅ |
| **Class A Predictions** | 20% → 55% ⚠️ SHIFTED! |
| **Ground Truth Labels** | Not available yet ❌ |

### The Challenge

Without ground truth labels, we cannot directly measure model accuracy. We must:
1. Identify **plausible causes** for the prediction shift
2. Run **diagnostic checks** to understand what's happening
3. Decide the **safest next action** (continue, pause, rollback, or route to review)

---

## 🔍 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| **Canary Rollout** | Gradual deployment strategy routing small % of traffic to new model |
| **Prediction Drift** | Change in the distribution of model predictions over time |
| **Covariate Shift** | Change in input data distribution from training to production |
| **PSI** | Population Stability Index - measures distribution stability |
| **KS-Test** | Kolmogorov-Smirnov test for comparing distributions |
| **Model Calibration** | How well predicted probabilities match actual outcomes |

---

## 💼 Interview Relevance

This scenario is a **common interview question** for:
- **ML Engineer** roles
- **MLOps Engineer** roles
- **Data Scientist** roles (production-focused)
- **ML Platform Engineer** roles

**Key interview takeaways:**
1. Never assume predictions are correct just because there are no errors
2. Use statistical tests (KS, PSI) to detect drift before labels are available
3. When in doubt, rollback or pause — user trust is more valuable than speed
4. Document your decision-making process for audit purposes

---

## 📚 Prerequisites

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Basic understanding of classification models

---

## 🎓 Author

Teaching Demo Project for Python, Data Science, and ML Learning

**Created:** 2026-01-16
