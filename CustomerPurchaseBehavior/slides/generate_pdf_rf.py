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

    # --- SLIDES CONTENT (RF) ---
    
    add_slide("Project: Customer Purchase Behavior (Random Forest)", [
        "Objective: High Accuracy Classification.",
        "Method: Random Forest Ensemble.",
        "Created by: AI Agent (Antigravity)"
    ])
    
    add_slide("Problem Statement", [
        "- Problem: Variability and Overfitting in single trees.",
        "- Goal: Use 'Wisdom of Crowds' for stability.",
        "- Context: Complex, noisy data."
    ])
    
    add_slide("Real-World Use Case", [
        "- Scenario: Credit Scoring / Recommendations.",
        "- Why RF? Industry standard for tabular data.",
        "- Reliability: Less likely to fail on edge cases."
    ])
    
    add_slide("Input Data", [
        "- Features: Demographics & Behavior.",
        "- Weights: Using 'balanced' weights.",
        "- Robustness: RF handles noisy inputs well."
    ])
    
    add_slide("Concepts Used", [
        "1. Ensemble Learning: Combining models.",
        "2. Bagging: Bootstrap Aggregating (Random sub-samples).",
        "3. Feature Randomness: Random features at split.",
        "4. Voting: Majority wins."
    ])
    
    add_slide("Concept: The Council of Experts", [
        "- Imagine 100 Experts.",
        "- Each expert sees a part of the data.",
        "- Expert 1 votes 'Sports'. Expert 2 votes 'Books'.",
        "- If 80 vote 'Sports', we go with Sports.",
        "- This cancels out individual mistakes."
    ])
    
    add_slide("Step-by-Step Flow", [
        "- Step 1: Bootstrap Data.",
        "- Step 2: Train 100 Trees.",
        "- Step 3: Aggregate Votes.",
        "- Step 4: Analyze Feature Importance.",
        "- Step 5: Evaluate."
    ])
    
    add_slide("Code Logic Summary", [
        "- Model: RandomForestClassifier(n_estimators=100, class_weight='balanced').",
        "- Attribute: feature_importances_."
    ])
    
    add_slide("Execution Results", [
        "- Overall Accuracy: ~94% (Excellent).",
        "- Recall: High across all classes.",
        "- Stability: Very robust."
    ])
    
    add_slide("Confusion Matrix", [
        "Visualizing performance:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/rf_confusion_matrix.png", 400, 350)
    ])
    
    add_slide("Feature Importance", [
        "What drives the decision?",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/rf_feature_importance.png", 600, 300)
    ])
    
    add_slide("Advantages & Limitations", [
        "- Pros: Very accurate, Handles imbalance/missing data well.",
        "- Cons: Slow training, large memory footprint, Black Box."
    ])
    
    add_slide("Interview Corner", [
        "- Q: Bagging vs Boosting? A: Bagging reduces variance (RF). Boosting reduces bias (XGB).",
        "- Q: OOB Score? A: Out-of-Bag score acts like a validation set built-in."
    ])
    
    add_slide("Conclusion", [
        "- Summary: Random Forest is likely our 'Champion Model'.",
        "- Next: Combine all findings."
    ])
    
    doc.build(story)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    create_pdf("C:/nagpython/demouv/CustomerPurchaseBehavior/slides/random_forest_slides.pdf")
