# TPR Drop Scanner Analysis - Teaching Project

## 🎯 Project Overview

This project explains and demonstrates the scenario where **slice-wise TPR (True Positive Rate) drops sharply for one site after a new scanner is introduced**, while service latency and error rate remain normal.

### Key Learning Objectives
1. Understand why this is a **data/model health issue** rather than a service issue
2. Learn **4+ diagnostics** to identify the root cause
3. Master **3+ mitigations** including safe-fallback options

---

## 📁 Project Structure

```
📁 tpr_drop_scanner_analysis/
│
├── 📁 notebook/
│   └── tpr_drop_scanner_analysis.ipynb    # Teaching-oriented notebook
│
├── 📁 documentation/
│   ├── problem_statement.md               # Problem definition
│   ├── concepts_explained.md              # Detailed concept explanations
│   └── observations_and_conclusion.md     # Results and insights
│
├── 📁 slides/
│   ├── notebooklm_style_slides.md         # 14-slide presentation
│   └── notebooklm_style_slides.html       # HTML version
│
├── 📁 src/
│   └── tpr_drop_analysis.py               # Source script
│
├── 📁 outputs/
│   ├── execution_output.md                # Captured output
│   └── sample_outputs/                    # Additional outputs
│
└── README.md                               # This file
```

---

## 📊 Generate PDF Slides

The slides are provided in HTML format for best viewing and printing to PDF.

### Step 1: Generate HTML Slides
```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe c:/nagpython/demouv/tpr_drop_scanner_analysis/slides/generate_slides_pdf.py
```

### Step 2: Convert to PDF
1. Open `slides/notebooklm_slides.html` in your browser
2. Press **Ctrl+P** (or Cmd+P on Mac)
3. Select **"Save as PDF"** as destination
4. Click **Save**

### Available Slide Files
- `notebooklm_style_slides.md` - Markdown source
- `notebooklm_slides.html` - NotebookLM-style HTML (recommended for PDF)
- `notebooklm_style_slides.html` - Dark theme HTML version
- `generate_slides_pdf.py` - Script to regenerate HTML

---

## 🚀 Quick Start

### Run the Analysis
```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe c:/nagpython/demouv/tpr_drop_scanner_analysis/src/tpr_drop_analysis.py
```

### Open the Notebook
```powershell
jupyter notebook c:/nagpython/demouv/tpr_drop_scanner_analysis/notebook/tpr_drop_scanner_analysis.ipynb
```

---

## 📋 Key Answers Summary

### 1. Why Data/Model Issue (Not Service Issue)?
- Service metrics (latency, error rate) are **normal** → infrastructure is healthy
- TPR drop is **isolated to one site/scanner** → data distribution shift
- New scanner produces images with **different characteristics** unseen during training
- Model makes **confident wrong predictions** (silent failure mode)

### 2. Diagnostics (Minimum 4)
| # | Diagnostic | Purpose |
|---|------------|---------|
| 1 | Feature Distribution Comparison | Detect KL divergence / histogram shift |
| 2 | Prediction Confidence Analysis | Check if model is confidently wrong |
| 3 | Per-Scanner Confusion Matrix | Identify misclassification patterns |
| 4 | Temporal TPR Trend Analysis | Confirm correlation with scanner rollout |

### 3. Mitigations (Minimum 3)
| Priority | Mitigation | Type |
|----------|------------|------|
| P0 | Route new scanner to human review | **Safe-Fallback** |
| P1 | Domain adaptation / fine-tuning | Model Fix |
| P2 | Preprocessing normalization | Data Fix |

---

## 💼 Interview Takeaways
- Always implement **slice-based monitoring** - overall accuracy hides subgroup issues
- **Covariate shift** can cause silent failures in production ML
- **Safe fallback** (human-in-the-loop) is critical for high-stakes applications
- Normal service metrics ≠ healthy model predictions

---

## 📚 Technologies Used
- Python 3.x
- NumPy, Pandas
- Scikit-learn
- SciPy (KL Divergence)

---

## 👤 Author
Teaching Demo - Educational Project for ML Monitoring

## 📅 Date
2026-01-16
