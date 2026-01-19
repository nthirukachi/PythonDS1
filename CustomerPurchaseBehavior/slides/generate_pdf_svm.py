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

    # --- SLIDES CONTENT (SVM) ---
    
    add_slide("Project: Customer Purchase Behavior (SVM)", [
        "Objective: Build a robust classifier using Support Vector Machines.",
        "Method: SVM with RBF Kernel.",
        "Created by: AI Agent (Antigravity)"
    ])
    
    add_slide("Problem Statement", [
        "- Problem: Automate customer categorization.",
        "- Context: 5000 records, 5 classes.",
        "- Challenge: Creating a decision boundary in complex, overlapping data."
    ])
    
    add_slide("Real-World Use Case", [
        "- Scenario: High-value customer targeting.",
        "- Application: Identify 'Gold' tier behaviors vs 'Basic'.",
        "- Impact: Precise targeting for marketing optimization."
    ])
    
    add_slide("Input Data", [
        "- Features: Age, Income, Monthly Spending, Sessions.",
        "- Processing: Scaled to mean 0, variance 1 (Critical for SVM)."
    ])
    
    add_slide("Concepts Used", [
        "1. Scaling: SVM is distance-based (margin).",
        "2. Hyperplane: The line separating classes.",
        "3. Kernel Trick (RBF): Bending space to separate non-linear data.",
        "4. Regularization (C): Balancing strictness vs smoothness."
    ])
    
    add_slide("SVM Concept (Simple)", [
        "- Imagine red and blue balls mixed on a table.",
        "- You want to separate them with a stick.",
        "- If you can't, you lift them into the air (3D).",
        "- Now you can slide a sheet between them.",
        "- That sheet is the Hyperplane. The lifting is the Kernel Trick."
    ])
    
    add_slide("Step-by-Step Flow", [
        "- Step 1: Load Data.",
        "- Step 2: Impute Missing Values.",
        "- Step 3: Scale Features (Essential!).",
        "- Step 4: Train SVM (RBF Kernel).",
        "- Step 5: Predict & Evaluate."
    ])
    
    add_slide("Data Visualization", [
        "Target Variable Distribution:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/svm_class_distribution.png", 500, 300)
    ])
    
    add_slide("Code Logic Summary", [
        "- Pipeline: Imputer -> Scaler -> Model.",
        "- Model: SVC(kernel='rbf', C=1.0, gamma='scale').",
        "- Fit on Train, Predict on Test."
    ])
    
    add_slide("Execution Results", [
        "- Overall Accuracy: ~75% (Higher than KNN's 69%).",
        "- Minority Class F1: 0.56 (Improved).",
        "- Confusion: Fewer mistakes between 'Fashion' and 'Electronics'."
    ])
    
    add_slide("Confusion Matrix", [
        "Visualizing model performance:",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/svm_confusion_matrix.png", 400, 350)
    ])
    
    add_slide("Observations & Insights", [
        "- Performance: SVM outperforms KNN clearly.",
        "- Reason: RBF kernel captures complex interactions.",
        "- Scaling: Without scaling, accuracy would drop significanty."
    ])
    
    add_slide("Advantages & Limitations", [
        "- Pros: High accuracy in high dimensions, Robust.",
        "- Cons: Slow to train on large datasets, Harder to interpret."
    ])
    
    add_slide("Conclusion", [
        "- Summary: SVM is a strong candidate (75% acc).",
        "- Recommendation: Deploy if inference speed isn't a bottleneck.",
        "- Note: Consider Random Forest for better interpretability."
    ])
    
    doc.build(story)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    create_pdf("C:/nagpython/demouv/CustomerPurchaseBehavior/slides/svm_slides.pdf")
