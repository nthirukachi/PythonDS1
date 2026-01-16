# ============================================================================
# Direct PDF Slide Generator - No Browser Required
# ============================================================================
# This script generates PDF slides directly using reportlab library.
#
# Usage:
#   python generate_pdf_direct.py
#
# Output:
#   - notebooklm_style_slides.pdf
# ============================================================================

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

OUTPUT_PDF = Path(__file__).parent / "notebooklm_style_slides.pdf"

# Color Scheme
COLORS = {
    'primary': colors.HexColor('#1a73e8'),
    'secondary': colors.HexColor('#34a853'),
    'accent': colors.HexColor('#ea4335'),
    'warning': colors.HexColor('#fbbc04'),
    'text': colors.HexColor('#202124'),
    'light_bg': colors.HexColor('#f8f9fa'),
    'white': colors.white,
}

# ----------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------

def get_styles():
    """Create custom styles for slides."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='SlideTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=COLORS['primary'],
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='SlideHeading',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=COLORS['primary'],
        spaceBefore=15,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='SlideSubHeading',
        parent=styles['Heading3'],
        fontSize=16,
        textColor=COLORS['text'],
        spaceBefore=10,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='SlideBody',
        parent=styles['Normal'],
        fontSize=14,
        textColor=COLORS['text'],
        spaceBefore=6,
        spaceAfter=6,
        leading=18,
        fontName='Helvetica'
    ))
    
    styles.add(ParagraphStyle(
        name='SlideBullet',
        parent=styles['Normal'],
        fontSize=13,
        textColor=COLORS['text'],
        leftIndent=20,
        spaceBefore=4,
        spaceAfter=4,
        bulletIndent=10,
        fontName='Helvetica'
    ))
    
    styles.add(ParagraphStyle(
        name='SlideNumber',
        parent=styles['Normal'],
        fontSize=10,
        textColor=COLORS['primary'],
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))
    
    return styles

# ----------------------------------------------------------------------------
# SLIDE CONTENT
# ----------------------------------------------------------------------------

SLIDES = [
    # Slide 1: Title
    {
        'title': 'TPR Drop Analysis After New Scanner Introduction',
        'content': [
            ('heading', 'Objective'),
            ('body', 'Understand why slice-wise TPR drops sharply for one site after a new scanner is introduced, while service metrics remain normal.'),
            ('heading', 'Learning Goals'),
            ('bullet', 'Distinguish data/model issues from service issues'),
            ('bullet', 'Master 4 diagnostic approaches'),
            ('bullet', 'Implement 3 mitigation strategies'),
        ]
    },
    # Slide 2: Problem Statement
    {
        'title': 'The Problem',
        'content': [
            ('body', 'Slice-wise TPR drops sharply for one site after a new scanner is introduced'),
            ('heading', 'Key Facts'),
            ('bullet', 'Before: Site 3 TPR = 0.92'),
            ('bullet', 'After: Site 3 TPR = 0.65 (DEGRADED)'),
            ('bullet', 'Service Latency: Normal (50ms)'),
            ('bullet', 'Error Rate: Normal (0.1%)'),
            ('heading', 'The Challenge'),
            ('body', 'System appears healthy, but predictions are wrong for Site 3!'),
        ]
    },
    # Slide 3: Real-World Use Case
    {
        'title': 'Real-World Use Case',
        'content': [
            ('heading', 'Medical Imaging Scenario'),
            ('bullet', '3 hospital sites with different scanners'),
            ('bullet', 'ML model for disease detection'),
            ('bullet', 'New scanner deployed at Site 3'),
            ('heading', 'The Risk'),
            ('bullet', 'False Negatives = Missed disease'),
            ('bullet', 'Patients at Site 3 receiving worse care'),
            ('bullet', 'System appears healthy but predictions are wrong'),
            ('heading', 'Industry Relevance'),
            ('body', 'Healthcare, Radiology, Pathology, and any ML system with diverse data sources'),
        ]
    },
    # Slide 4: Input Data
    {
        'title': 'Input Data / Inputs',
        'content': [
            ('heading', 'Multi-Site Scanner Data'),
            ('table', [
                ['Scanner', 'Site', 'Mean', 'Std', 'Type'],
                ['Scanner_A', 'Site_1', '100', '15', 'Original'],
                ['Scanner_B', 'Site_2', '100', '15', 'Original'],
                ['Scanner_C', 'Site_3', '120', '25', 'NEW'],
            ]),
            ('heading', 'Key Difference'),
            ('body', 'New scanner has DIFFERENT distribution (Mean: 120 vs 100, Std: 25 vs 15)'),
        ]
    },
    # Slide 5: Concepts Used
    {
        'title': 'Concepts Used (High Level)',
        'content': [
            ('table', [
                ['Concept', 'Purpose'],
                ['TPR (Recall)', 'Measure positive class detection rate'],
                ['Covariate Shift', 'Input distribution change'],
                ['Slice-based Monitoring', 'Per-subgroup evaluation'],
                ['KL Divergence', 'Quantify distribution difference'],
                ['Safe Fallback', 'Human-in-the-loop protection'],
                ['Domain Adaptation', 'Model robustness improvement'],
            ]),
        ]
    },
    # Slide 6: Simple Explanations
    {
        'title': 'Concepts Breakdown (Simple)',
        'content': [
            ('heading', 'TPR = True Positive Rate'),
            ('body', '"Out of 100 disease cases, how many did we correctly detect?"'),
            ('heading', 'Covariate Shift'),
            ('body', '"Training data looks different from production data"'),
            ('heading', 'Slice-based Monitoring'),
            ('body', '"Check accuracy for each group separately, not just overall"'),
            ('heading', 'Safe Fallback'),
            ('body', '"When unsure, let a human expert decide"'),
        ]
    },
    # Slide 7: Solution Flow
    {
        'title': 'Step-by-Step Solution Flow',
        'content': [
            ('heading', '1. OBSERVE'),
            ('bullet', 'TPR drops for Site 3'),
            ('bullet', 'Service metrics remain normal'),
            ('heading', '2. DIAGNOSE'),
            ('bullet', 'Feature distribution comparison'),
            ('bullet', 'Confidence calibration check'),
            ('bullet', 'Confusion matrix breakdown'),
            ('bullet', 'Temporal trend analysis'),
            ('heading', '3. MITIGATE'),
            ('bullet', 'P0: Safe fallback (human review)'),
            ('bullet', 'P1: Preprocessing normalization'),
            ('bullet', 'P2: Domain adaptation / retraining'),
        ]
    },
    # Slide 8: Code Logic
    {
        'title': 'Code Logic Summary',
        'content': [
            ('heading', 'Main Sections'),
            ('bullet', '1. Simulate Data: 3 scanners with different distributions'),
            ('bullet', '2. Train Model: Only on original scanners (A, B)'),
            ('bullet', '3. Evaluate per Slice: TPR for each scanner'),
            ('bullet', '4. Check Service Metrics: Confirm infrastructure healthy'),
            ('bullet', '5. Run Diagnostics: 4 diagnostic functions'),
            ('bullet', '6. Apply Mitigations: 3 mitigation strategies'),
            ('heading', 'Key Functions'),
            ('body', 'simulate_scanner_data(), train_model_on_original_scanners(), evaluate_per_scanner(), diagnostic_*(), mitigation_*()'),
        ]
    },
    # Slide 9: Important Functions
    {
        'title': 'Important Functions & Parameters',
        'content': [
            ('heading', 'evaluate_per_scanner(df, model, scaler)'),
            ('body', 'Calculates TPR per scanner slice, returns confusion matrix breakdown'),
            ('heading', 'diagnostic_feature_distribution(df)'),
            ('body', 'Computes KL divergence per feature. Threshold > 0.1 = significant shift'),
            ('heading', 'mitigation_safe_fallback(df, model, scaler, confidence_threshold=0.7)'),
            ('body', 'Routes low-confidence predictions to human review. Default threshold: 0.7'),
        ]
    },
    # Slide 10: Execution Output
    {
        'title': 'Execution Output',
        'content': [
            ('heading', 'TPR per Scanner'),
            ('table', [
                ['Scanner', 'TPR', 'Status'],
                ['Scanner_A', '0.91', 'Normal'],
                ['Scanner_B', '0.90', 'Normal'],
                ['Scanner_C_NEW', '0.65', 'DEGRADED'],
            ]),
            ('heading', 'Diagnostic: KL Divergence'),
            ('body', 'feature_0: 0.55 (significant shift detected!)'),
            ('body', 'feature_1: 1.21 (HIGH distribution shift!)'),
        ]
    },
    # Slide 11: Observations
    {
        'title': 'Observations & Insights',
        'content': [
            ('heading', '1. Overall Accuracy Hides Subgroup Issues'),
            ('body', 'If we only monitored overall accuracy, we would NOT have detected this issue'),
            ('heading', '2. Normal Service != Healthy Model'),
            ('body', 'System appears healthy but predictions are wrong (silent failure mode)'),
            ('heading', '3. New Scanner = Distribution Shift'),
            ('body', 'KL Divergence confirms the new scanner data is very different'),
            ('heading', '4. Model is Confidently Wrong'),
            ('body', 'High confidence on false negatives = miscalibration'),
        ]
    },
    # Slide 12: Advantages & Limitations
    {
        'title': 'Advantages & Limitations',
        'content': [
            ('heading', 'Advantages'),
            ('bullet', 'Detects silent failures'),
            ('bullet', 'Quantifies distribution shift'),
            ('bullet', 'Provides actionable mitigations'),
            ('bullet', 'Enables safe deployment'),
            ('heading', 'Limitations'),
            ('bullet', 'Requires labeled data for adaptation'),
            ('bullet', 'KL divergence sensitive to binning'),
            ('bullet', 'Human review does not scale infinitely'),
            ('bullet', 'Preprocessing may not capture all differences'),
        ]
    },
    # Slide 13: Interview Takeaways
    {
        'title': 'Interview Key Takeaways',
        'content': [
            ('heading', 'Q1: Why is this a data/model issue, not service issue?'),
            ('body', 'Service metrics are normal. TPR drop is isolated to one site/scanner, pointing to data distribution shift.'),
            ('heading', 'Q2: What diagnostics would you run?'),
            ('body', 'KL divergence, confidence calibration, per-scanner confusion matrices, temporal trend analysis.'),
            ('heading', 'Q3: What is your first mitigation?'),
            ('body', 'Safe fallback - route low-confidence predictions to human review for immediate protection.'),
        ]
    },
    # Slide 14: Conclusion
    {
        'title': 'Conclusion',
        'content': [
            ('heading', 'Root Cause'),
            ('body', 'Covariate shift from new scanner with different data characteristics'),
            ('heading', 'Evidence'),
            ('bullet', 'Service metrics normal - Infrastructure healthy'),
            ('bullet', 'TPR drop isolated to Site 3 - Not global failure'),
            ('bullet', 'High KL divergence - Distribution shift confirmed'),
            ('bullet', 'Temporal correlation - New scanner is cause'),
            ('heading', 'Action Plan'),
            ('bullet', 'P0: Safe fallback (human review)'),
            ('bullet', 'P1: Preprocessing normalization'),
            ('bullet', 'P2: Domain adaptation / retraining'),
            ('heading', 'Key Takeaway'),
            ('body', 'Always monitor slice-wise metrics. Service health != model health. Implement safe fallbacks for high-stakes applications.'),
        ]
    },
]

# ----------------------------------------------------------------------------
# PDF GENERATION
# ----------------------------------------------------------------------------

def create_table(data, styles):
    """Create a styled table."""
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), COLORS['light_bg']),
        ('TEXTCOLOR', (0, 1), (-1, -1), COLORS['text']),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dadce0')),
    ]))
    return table

def generate_slide(slide_num, slide_data, styles):
    """Generate content for a single slide."""
    elements = []
    
    # Add slide number
    elements.append(Paragraph(f"Slide {slide_num} of {len(SLIDES)}", styles['SlideNumber']))
    elements.append(Spacer(1, 10))
    
    # Add title
    elements.append(Paragraph(slide_data['title'], styles['SlideTitle']))
    elements.append(Spacer(1, 20))
    
    # Add content
    for item_type, item_content in slide_data['content']:
        if item_type == 'heading':
            elements.append(Paragraph(item_content, styles['SlideHeading']))
        elif item_type == 'subheading':
            elements.append(Paragraph(item_content, styles['SlideSubHeading']))
        elif item_type == 'body':
            elements.append(Paragraph(item_content, styles['SlideBody']))
        elif item_type == 'bullet':
            elements.append(Paragraph(f"  * {item_content}", styles['SlideBullet']))
        elif item_type == 'table':
            elements.append(Spacer(1, 10))
            elements.append(create_table(item_content, styles))
            elements.append(Spacer(1, 10))
    
    # Add page break
    elements.append(PageBreak())
    
    return elements

def generate_pdf():
    """Generate the complete PDF."""
    print("[START] Generating PDF slides...")
    
    # Create document
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(LETTER),
        rightMargin=50,
        leftMargin=50,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = get_styles()
    elements = []
    
    # Generate each slide
    for i, slide in enumerate(SLIDES, 1):
        print(f"[SLIDE] Generating slide {i}: {slide['title']}")
        elements.extend(generate_slide(i, slide, styles))
    
    # Build PDF
    doc.build(elements)
    
    print(f"\n[OK] PDF generated successfully: {OUTPUT_PDF}")
    print(f"[INFO] Total slides: {len(SLIDES)}")

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TPR Drop Analysis - Direct PDF Generator")
    print("="*60 + "\n")
    
    generate_pdf()
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60 + "\n")
