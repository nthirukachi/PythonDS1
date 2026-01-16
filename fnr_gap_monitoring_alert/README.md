# 🚨 FNR Gap Monitoring Alert System

## 📋 Project Overview

This project implements a **False Negative Rate (FNR) Gap Monitoring Alert System** designed to detect **unequal harm** across different operational slices in deployed binary classifiers.

### What is this about?

In real-world machine learning deployments (like medical diagnosis or fraud detection), different groups of users may experience different error rates. This system monitors for such disparities by tracking the **FNR Gap** - the difference between the worst and best performing slices.

### Key Formula

```
FNR_g = FN_g / (TP_g + FN_g)

Gap = max_g(FNR_g) - min_g(FNR_g)
```

Where:
- `g` = operational slice (e.g., device type, hospital site)
- `FN_g` = False Negatives for slice g
- `TP_g` = True Positives for slice g
- `FNR_g` = False Negative Rate for slice g

---

## 📁 Folder Structure

```
📁 fnr_gap_monitoring_alert/
│
├── 📁 notebook/
│   └── fnr_gap_monitoring_alert.ipynb    # Teaching-oriented Jupyter Notebook
│
├── 📁 documentation/
│   ├── problem_statement.md              # Problem definition and approach
│   ├── concepts_explained.md             # Detailed concept explanations
│   └── observations_and_conclusion.md    # Results and insights
│
├── 📁 slides/
│   └── notebooklm_style_slides.md        # 14-slide presentation
│
├── 📁 src/
│   └── fnr_gap_monitoring.py             # Main Python script
│
├── 📁 outputs/
│   ├── execution_output.md               # Captured outputs
│   └── sample_outputs/                   # Sample output files
│
└── README.md                             # This file
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `matplotlib`

### Running the Python Script
```powershell
cd c:\nagpython\demouv\fnr_gap_monitoring_alert\src
python fnr_gap_monitoring.py
```

### Running the Jupyter Notebook
```powershell
cd c:\nagpython\demouv\fnr_gap_monitoring_alert\notebook
jupyter notebook fnr_gap_monitoring_alert.ipynb
```

---

## 🎯 Key Concepts

| Concept | Description |
|---------|-------------|
| **FNR (False Negative Rate)** | Proportion of actual positives incorrectly classified as negative |
| **Operational Slice** | A subgroup of data (e.g., device type, hospital site) |
| **Gap Metric** | Difference between worst and best FNR across slices |
| **Alert Threshold** | Gap value that triggers an alert (e.g., > 0.10) |
| **Time Window** | Period over which metrics are aggregated (e.g., 4 weeks) |
| **Runbook** | Documented steps to follow when an alert fires |

---

## 📚 Learning Objectives

After studying this project, you will understand:

1. ✅ How to calculate FNR for different data slices
2. ✅ How to design fairness monitoring metrics
3. ✅ How to set appropriate alert thresholds
4. ✅ How to create runbooks for ML monitoring
5. ✅ Real-world applications in responsible AI

---

## 🏥 Real-World Use Cases

- **Healthcare**: Detecting if a diagnostic model misses more cases for certain hospital sites
- **Finance**: Monitoring if fraud detection fails more for certain device types
- **Insurance**: Checking if claim approval differs across demographic groups

---

## 👨‍💻 Author

Created as a teaching project for learning Responsible AI and ML Monitoring concepts.

---

## 📄 License

Educational use only.
