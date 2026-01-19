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

    # --- SLIDES CONTENT (DT) ---
    
    add_slide("Project: Customer Purchase Behavior (Decision Tree)", [
        "Objective: Explainable Classification.",
        "Method: Decision Tree with Class Balancing.",
        "Created by: AI Agent (Antigravity)"
    ])
    
    add_slide("Problem Statement", [
        "- Problem: 'Black Box' models are hard to trust.",
        "- Goal: Explain WHY a customer is classified as 'Sports'.",
        "- Context: Regulatory environments often mandate explainability."
    ])
    
    add_slide("Real-World Use Case", [
        "- Scenario: Loan Approvals / Fraud.",
        "- Reason Code: 'Denied because Income < X and Debt > Y'.",
        "- E-commerce: Debugging why a campaign failed."
    ])
    
    add_slide("Input Data", [
        "- Features: Standard (Age, Income, etc.).",
        "- Scaling: Not strictly needed for Trees but used for consistency.",
        "- Weights: Using 'balanced' weights to help minority classes."
    ])
    
    add_slide("Concepts Used", [
        "1. Nodes/Leaves: Questions and Answers.",
        "2. Gini Impurity: Metric for splitting.",
        "3. Max Depth: Pruning to prevent overfitting.",
        "4. Interpretability: White-box modeling."
    ])
    
    add_slide("Concept: 20 Questions", [
        "- Think of a Decision Tree as a game of 20 Questions.",
        "- Q1: Is spending > 500? (Yes/No)",
        "- Q2: Is Age > 30? (Yes/No)",
        "- Answer: Electronics Category."
    ])
    
    add_slide("Step-by-Step Flow", [
        "- Step 1: Load & Clean.",
        "- Step 2: Configure Tree (max_depth=4, class_weight='balanced').",
        "- Step 3: Train.",
        "- Step 4: Visualise Logic.",
        "- Step 5: Evaluate."
    ])
    
    add_slide("Code Logic Summary", [
        "- Model: DecisionTreeClassifier(max_depth=4, class_weight='balanced').",
        "- Visualization: plot_tree()."
    ])
    
    add_slide("Execution Results", [
        "- Overall Accuracy: (Check Code Output).",
        "- Recall: Should be better for Class 4 (Sports) due to weights.",
        "- Trade-off: Lower precision potentially."
    ])
    
    add_slide("Confusion Matrix", [
        "Visualizing performance:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/dt_confusion_matrix.png", 400, 350)
    ])
    
    add_slide("Tree Visualization", [
        "The logic map:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/dt_visualization.png", 600, 300)
    ])
    
    add_slide("Advantages & Limitations", [
        "- Pros: Explainable, handles non-linearities, no scaling needed.",
        "- Cons: Overfitting (if deep), Instability (high variance)."
    ])
    
    add_slide("Interview Corner", [
        "- Q: Gini vs Entropy? A: Similar results, Gini is faster.",
        "- Q: Pruning? A: Limiting depth to improve generalization."
    ])
    
    add_slide("Conclusion", [
        "- Summary: We have an explainable model.",
        "- Next: Random Forest to average out the variance of a single tree."
    ])
    
    doc.build(story)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    create_pdf("C:/nagpython/demouv/CustomerPurchaseBehavior/slides/decision_tree_slides.pdf")
