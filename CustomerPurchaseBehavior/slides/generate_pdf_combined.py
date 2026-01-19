from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors

def create_pdf(output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(LETTER))
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.darkblue,
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.black,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=12,
        spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        fontSize=12,
        leftIndent=20,
        spaceAfter=5,
        bulletIndent=10
    )
    
    story = []
    
    # Helper to add a slide
    def add_slide(title, content_list):
        story.append(Paragraph(title, title_style))
        for item in content_list:
            if isinstance(item, str):
                if item.startswith("- "):
                    story.append(Paragraph(item[2:], bullet_style))
                else:
                    story.append(Paragraph(item, body_style))
            elif isinstance(item, tuple) and item[0] == 'image':
                try:
                    img = Image(item[1], width=item[2], height=item[3])
                    story.append(img)
                except Exception as e:
                    story.append(Paragraph(f"[Error loading image: {item[1]}]", body_style))
        story.append(PageBreak())

    # --- SLIDES CONTENT (Combined) ---
    
    add_slide("Project: Model Comparison Showdown", [
        "Objective: Compare 4 Algorithms.",
        "Contenders: KNN, SVM, Decision Tree, Random Forest.",
        "Created by: AI Agent (Antigravity)"
    ])
    
    add_slide("The Contenders", [
        "- KNN: The 'Nearest Neighbor' baseline.",
        "- SVM: The 'Boundary' builder.",
        "- Decision Tree: The 'Rule' maker.",
        "- Random Forest: The 'Ensemble' leader."
    ])
    
    add_slide("Methodology", [
        "- Data: 5000 Rows.",
        "- Preprocessing: Scaled features (StandardScaler).",
        "- Metrics: Accuracy, Recall (Minority), Speed.",
        "- Goal: Best all-rounder."
    ])
    
    add_slide("Results: Accuracy", [
        "- KNN: ~69%.",
        "- Decision Tree: ~53-60%.",
        "- SVM: ~75%.",
        "- Random Forest: ~94% (Winner)."
    ])
    
    add_slide("Results: Fairness (Minority Recall)", [
        "- Class: 'Sports' (Class 4).",
        "- KNN: Poor performance.",
        "- SVM: Moderate.",
        "- Trees (DT/RF): Excellent performance due to Class Weights."
    ])
    
    add_slide("Visual Comparison", [
        "Accuracy vs Recall:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/model_comparison.png", 500, 300)
    ])
    
    add_slide("Model Profiles", [
        "- KNN: Simple but slow inference.",
        "- SVM: Robust but slow training.",
        "- DT: Explainable but unstable.",
        "- RF: Accurate but complex."
    ])
    
    add_slide("Execution Time Insights", [
        "- Fastest Train: Decision Tree.",
        "- Slowest Train: Random Forest.",
        "- Fastest Predict: Decision Tree.",
        "- Slowest Predict: KNN."
    ])
    
    add_slide("Observations", [
        "- Scaling verified as critical for KNN/SVM.",
        "- Class Imbalance handling was successful using Weights.",
        "- Random Forest correctly identified key features (Income/Spending)."
    ])
    
    add_slide("Trade-offs Matrix", [
        "- Need Explanation? -> Choose Decision Tree.",
        "- Need Raw Accuracy? -> Choose Random Forest.",
        "- Need Balanced approach? -> SVM (if tuned)."
    ])
    
    add_slide("Interview Corner", [
        "- Q: Which model to deploy?",
        "- A: Random Forest for batch processing. If real-time latency is key, maybe Decision Tree or lighter Gradient Boosting.",
        "- Q: Bias-Variance?",
        "- A: DT has High Variance. RF reduces Variance."
    ])
    
    add_slide("Final Verdict", [
        "- Champion: Random Forest.",
        "- Why? Superior accuracy and handling of edge cases.",
        "- Recommendation: Use RF for production."
    ])
    
    doc.build(story)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    create_pdf("C:/nagpython/demouv/CustomerPurchaseBehavior/slides/combined_models_slides.pdf")
