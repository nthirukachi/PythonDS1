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
                    story.append(Paragraph(item[2:], bullet_style)) # Bullet
                else:
                    story.append(Paragraph(item, body_style)) # Text
            elif isinstance(item, tuple) and item[0] == 'image':
                # (image, path, width, height)
                try:
                    img = Image(item[1], width=item[2], height=item[3])
                    story.append(img)
                except Exception as e:
                    story.append(Paragraph(f"[Error loading image: {item[1]}]", body_style))
        story.append(PageBreak())

    # --- SLIDES CONTENT ---
    
    # Slide 1
    add_slide("Project: Customer Purchase Behavior (KNN)", [
        "Objective: Build a model to classify customers into purchase categories.",
        "Method: K-Nearest Neighbors (KNN) Algorithm.",
        "Created by: AI Agent (Antigravity)"
    ])
    
    # Slide 2
    add_slide("Problem Statement", [
        "- Problem: Need to automate customer segmentation.",
        "- Context: 5000 records, 5 categories (Electronics to Sports).",
        "- Challenge: Significant class imbalance and missing data."
    ])
    
    # Slide 3
    add_slide("Real-World Use Case", [
        "- Scenario: Personalized homepage recommendations.",
        "- Application: Recommend 'Running Shoes' to 'Sports' users.",
        "- Impact: Increased revenue through targeted marketing."
    ])
    
    # Slide 4
    add_slide("Input Data", [
        "- Demographics: Age, Income, Account Age.",
        "- Behavior: Spending, Session Duration, Page Views.",
        "- Categorical: Device Type, Membership Tier.",
        "- Target: Purchase Category (0-4)."
    ])
    
    # Slide 5
    add_slide("Concepts Used", [
        "1. Imputation: Filling missing values (Mean).",
        "2. Encoding: Converting text to numbers (OneHot).",
        "3. Scaling: Normalizing features (StandardScaler).",
        "4. KNN Algorithm: Distance-based classification."
    ])
    
    # Slide 6
    add_slide("Step-by-Step Flow", [
        "- Step 1: Load Data.",
        "- Step 2: Clean Data (Impute & Encode).",
        "- Step 3: Scale Features (Crucial for KNN).",
        "- Step 4: Train KNN (k=5).",
        "- Step 5: Evaluate on Test Set."
    ])
    
    # Slide 7: Visualization (Class Distribution)
    add_slide("Data Visualization: Class Distribution", [
        "Note the imbalance towards Class 0 (Electronics).",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/class_distribution.png", 500, 300)
    ])
    
    # Slide 8: Code Logic Summary
    add_slide("Code Logic Summary", [
        "- Pipeline: Imputer -> Scaler -> Model.",
        "- Split: 70% Train, 30% Test (Stratified).",
        "- Model: KNeighborsClassifier(n_neighbors=5)."
    ])
    
    # Slide 9: Execution Output & Metrics
    add_slide("Execution Results", [
        "- Overall Accuracy: ~69%",
        "- Best Class: Electronics (High Recall).",
        "- Worst Class: Home & Sports (Low Recall due to less data)."
    ])
    
    # Slide 10: Confusion Matrix
    add_slide("Confusion Matrix", [
        "Visualizing where the model makes mistakes.",
        ("image", "C:/nagpython/demouv/CustomerPurchaseBehavior/outputs/sample_outputs/knn_confusion_matrix.png", 400, 350)
    ])
    
    # Slide 11: Observations
    add_slide("Observations & Insights", [
        "- Imbalance Impact: Model favors majority class.",
        "- Feature Sensitivity: Scaling was necessary for KNN.",
        "- Missing Data: ~1% of rows had missing values, imputation handled this."
    ])
    
    # Slide 12: Advantages & Limitations
    add_slide("Advantages & Limitations (KNN)", [
        "- Pros: Simple, Intuitive, No training time.",
        "- Cons: Slow prediction on large data, Sensitive to outliers."
    ])
    
    # Slide 13: Interview Takeaways
    add_slide("Interview Corner", [
        "- Q: Why scale for KNN? A: Distance metrics are sensitive to magnitude.",
        "- Q: How to handle imbalance? A: SMOTE, or adjust Class Weights (not available in standard KNN, use distance weights)."
    ])
    
    # Slide 14: Conclusion
    add_slide("Conclusion", [
        "- We established a baseline with KNN (69%).",
        "- Next steps: Try Random Forest to handle non-linearity better.",
        "- Recommendation: Collect more data for 'Sports' category."
    ])
    
    doc.build(story)
    print(f"PDF generated: {output_filename}")

if __name__ == "__main__":
    create_pdf("C:/nagpython/demouv/CustomerPurchaseBehavior/slides/knn_slides.pdf")
