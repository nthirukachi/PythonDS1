---
description: How to generate PDF slides for teaching projects (NotebookLM Style)
---

# PDF Slide Generation Workflow (NotebookLM Style)

## IMPORTANT: Always Use Direct PDF Generation
**DO NOT** use browser print-to-PDF approach.
**ALWAYS** generate PDF directly using Python `reportlab`.

## Method: Use reportlab + Matplotlib

### Step 1: Ensure libraries are installed
```powershell
& c:/nagpython/demouv/.venv/Scripts/pip.exe install reportlab matplotlib seaborn
```

### Step 2: Create a Diagram Generator
Create `slides/generate_diagrams.py` to produce clean, "NotebookLM-style" visualizations (Pastel colors, minimalist).
- **Distribution Plots**: For Data Drift (P(X)).
- **Sigmoid/Scatter Plots**: For Concept Drift (P(Y|X)).
- **Flowcharts**: For System Architecture (using `matplotlib.patches`).
- **Output**: Save all images to `slides/images/*.png`.

### Step 3: Create PDF Generator
Create `slides/generate_pdf_direct.py` with:
- `reportlab` imports (`SimpleDocTemplate`, `Image`, etc.).
- Defined `SLIDES` list with structured content.
- **Image Embedding**: Import and resize images from `slides/images/` into relevant slides.
- Use landscape LETTER page size.

### Step 4: Run the Generators
```powershell
# 1. Generate Images
& c:/nagpython/demouv/.venv/Scripts/python.exe path/to/slides/generate_diagrams.py

# 2. Generate PDF
& c:/nagpython/demouv/.venv/Scripts/python.exe path/to/slides/generate_pdf_direct.py
```

### Step 5: Cleanup (MANDATORY)
Once the PDF is generated and verified:
**DELETE** the generator scripts to keep the project clean.
```powershell
del path/to/slides/generate_diagrams.py
del path/to/slides/generate_pdf_direct.py
# Also delete notebook generator if applicable
del path/to/notebook/create_notebook.py
```

## Template Code Structure
```python
from reportlab.platypus import Image
# ...
def get_image(filename):
    return Image(path, width=400, height=200, kind='proportional')
# ...
slides = [
    {"title": "Slide With Image", "content": [("image", "my_plot.png")]}
]
```
