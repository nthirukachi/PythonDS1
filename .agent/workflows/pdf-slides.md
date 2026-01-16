---
description: How to generate PDF slides for teaching projects
---

# PDF Slide Generation Workflow

## IMPORTANT: Always Use Direct PDF Generation

**DO NOT** use browser print-to-PDF approach. 
**ALWAYS** generate PDF directly using Python libraries.

## Method: Use reportlab Library

### Step 1: Ensure reportlab is installed
```powershell
& c:/nagpython/demouv/.venv/Scripts/pip.exe install reportlab
```

### Step 2: Create a direct PDF generator script

Create `generate_pdf_direct.py` in the `slides/` folder with:
- Import reportlab components
- Define slide content as structured data
- Use landscape LETTER page size
- Create styled paragraphs, headings, tables, and bullets
- Generate PDF directly without browser

### Step 3: Run the generator
```powershell
& c:/nagpython/demouv/.venv/Scripts/python.exe path/to/slides/generate_pdf_direct.py
```

## Template Code Structure

```python
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Define SLIDES as list of dicts with 'title' and 'content'
# Content is list of tuples: ('heading'|'body'|'bullet'|'table', content)

def generate_pdf():
    doc = SimpleDocTemplate("slides.pdf", pagesize=landscape(LETTER))
    # ... generate elements from SLIDES
    doc.build(elements)
```

## Reference Implementation

See: `c:\nagpython\demouv\tpr_drop_scanner_analysis\slides\generate_pdf_direct.py`

## Why Direct PDF?
1. No manual browser interaction required
2. Consistent output every time
3. Can be automated in scripts
4. Works in headless environments
