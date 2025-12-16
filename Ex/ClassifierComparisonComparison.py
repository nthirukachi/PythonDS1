"""
Problem Statement:
Classifier Comparison and Analysis for Medical Diagnosis.
Context:
- Dataset: Medical diagnosis dataset.
- Constraints: 
    1. Critical Cost: False Negatives (Missing disease). -> High Recall is Priority #1.
    2. Throughput: 1000+ patients/hour. -> Latency < 3.6s per prediction is fine, but faster is better for scalability.
    3. Interpretability: Doctors need to understand "Why".
- Current Performance Metrics provided in prompt.

Task:
1. Analyze most important metrics.
2. Rank classifiers.
3. Identify the best classifier.
4. Suggest modification for the second-best.

Solution approach:
- Creating a Python script that defines the metrics data structure.
- Calculating a weighted score for each classifier based on the business requirements (Recall > Interpretability > Speed > Precision).
- Printing the detailed analysis texts as requested.

Expected Output:
- Textual answers to Parts 1, 2, 3, 4 with technical justification based on the provided data table.


Based on the analysis of your metrics table and constraints, here are the answers:

1. Important Metrics Analysis
Primary Metric: Recall (Sensitivity)

Why: The problem states "Missing a disease (False Negative) is critical and costly." High Recall ensures that almost every sick patient is identified. A Recall of 0.78 (k-NN) means missing 22% of sick people, which is unacceptable. You need the highest possible Recall (0.89+).
Secondary Metric: Interpretability

Why: "Model interpretability is important for doctors." Medical professionals rarely trust a "Black Box" algorithm. They need to know why a diagnosis was made (e.g., "Age > 50" vs "Complex Kernel Vector").
(Note on Speed): While "1000 patients/hour" sounds high, it is only ~0.3 predictions per second. Even the slowest model (k-NN at 2.3s) assumes a single thread; in parallel batch processing, speed is likely not the bottleneck compared to accuracy and trust.

2. Classifier Ranking
Logistic Regression (Best Balance: High Recall + Transparent)
SVM (RBF Kernel) (Best Performance: Highest Recall 0.92, but lacks transparency)
Decision Tree (Good Transparency, but mediocre Recall 0.85)
k-NN (Unsuitable: Lowest Recall 0.78 and Slowest)
3. Best Classifier Identification
Choice: Logistic Regression

Justification:

Safety (Recall 0.89): It misses very few cases, performing nearly as well as the much more complex SVM (0.92).
Trust (Interpretability): It provides Odds Ratios. A doctor can look at the model weights and say "This patient is flagged because their Blood Pressure contributes X% to the risk." This transparency is non-negotiable in many clinical settings.
Deployment: It is lightning fast (0.01s), easily handling the throughput requirement.
4. Modification for Second-Best Classifier
Classifier: Support Vector Machine (RBF Kernel).

Weakness: It is a "Black Box" (Low Interpretability).
Modification: Implement SHAP (SHapley Additive exPlanations) or LIME.
How it works: You keep the high-performance RBF model (0.92 Recall) running. You then use the SHAP wrapper to generate a "Reason Code" for every prediction (e.g., "Feature A pushed the probability up by 20%").
Result: This essentially gives you the "best of both worlds"—the accuracy of the SVM with the explainability of the Logistic Regression.

"""

# Why: Import Pandas for easy table display and sorting of the metrics.
# Output: Module 'pandas' loaded.
import pandas as pd

# ==========================================
# 1. Comparison Matrix Setup
# ==========================================

# Define Data
# What: A dictionary representing the performance table provided in the problem statement.
# Why: To programmatically analyze and display the options.
data = {
    'Classifier': ['Logistic Regression', 'k-NN (k=5)', 'SVM (RBF)', 'Decision Tree'],
    'Accuracy': [0.87, 0.85, 0.91, 0.84],
    'Precision': [0.83, 0.88, 0.89, 0.81],
    'Recall': [0.89, 0.78, 0.92, 0.85],
    'F1-Score': [0.86, 0.83, 0.90, 0.83],
    'Training Time (s)': [0.5, 0, 15, 1.2],
    'Prediction Time (s)': [0.01, 2.3, 0.8, 0.02],
    'Interpretability': ['High (Coefficients)', 'Low (Black Box)', 'Low (Kernel/Black Box)', 'High (Rules)'] 
}

# Create DataFrame
# Output: A structured table.
df = pd.DataFrame(data)

print("--- Provided Metrics Table ---")
print(df.to_string(index=False))
print("\n" + "="*50 + "\n")

# ==========================================
# Task 1: Analyze Important Metrics
# ==========================================

print("--- Task 1: Metric Analysis ---")
print("Question: Which metric(s) are most important for this use case and why?")
print("""
Answer:
1. Recall (Most Critical):
   Why: The problem states "Missing a disease (False Negative) is critical and costly." 
   Recall = TP / (TP + FN). Maximizing Recall minimizes FN. Ideally, we want >0.95, but 0.92 is the best available.

2. Interpretability (Secondary Critical):
   Why: "Model interpretability is important for doctors." 
   Medical AI is often a decision-support tool. If a doctor can't trust the logic, they won't use it.

3. Prediction Time (Constraint):
   Why: "1000+ patients per hour" = ~3.6 seconds per patient.
   All models fit this (slowest is k-NN at 2.3s), but lower latency is better for batch processing.
""")

# ==========================================
# Task 2: Rank Classifiers
# ==========================================

print("--- Task 2: Classifier Ranking ---")
print("Question: Rank the classifiers based on suitability.")
print("""
RANKING LOGIC:
- SVM: Winner on Recall (0.92) and Accuracy (0.91), but fails Interpretability.
- LogReg: Strong Recall (0.89), Very Fast (0.01s), High Interpretability.
- Decision Tree: Good Interpretability, okay Recall (0.85), but lowest Accuracy (0.84).
- k-NN: Loser. Lowest Recall (0.78), Slowest (2.3s).

FINAL DECISION (Considering Recall + Interpretability balance):
1. Logistic Regression (The Best Compromise)
2. SVM (The Performance King, if explainability can be solved)
3. Decision Tree (Explainable, but weak performance)
4. k-NN (Unusable due to low Recall)
""")

# ==========================================
# Task 3: Identify Best Classifier
# ==========================================

print("--- Task 3: Best Classifier Selection ---")
print("Question: Identify the best classifier and justify.")
print("""
Selection: Logistic Regression.

Justification:
1. High Recall (0.89): It captures 89% of sick patients, which is close to the SVM's 0.92.
2. High Interpretability: It provides "Odds Ratios". A doctor can see "Age coefficient is 0.5", meaning older age increases risk. This satisfies the "Interpretability" constraint which SVM fails.
3. Speed (0.01s): It processes 1,000 patients in just 10 seconds (vs SVM's 13 minutes or k-NN's 38 minutes).
4. Safety: While SVM is slightly more accurate, the "Black Box" nature makes it risky for liability. LogReg is transparent.
""")

# ==========================================
# Task 4: Suggest Modification
# ==========================================

print("--- Task 4: Modification for 2nd Best ---")
print("Question: Suggest one specific modification to improve the second-best classifier.")
print("Second Best chosen: SVM (RBF)")
print("""
Problem with SVM: It is a Black Box (Hard to interpret) and computationally heavy.

Modification: Use SHAP (SHapley Additive exPlanations) or LIME.

Explanation:
- Implementation: Train the RBF SVM as usual to keep the 0.92 Recall.
- Wrapper: Apply SHAP KernelExplainer on top of the trained SVM.
- Output: SHAP will generate a "Feature Importance" plot for individual predictions (e.g., "Patient flagged due to Glucose > 180").
- Why: This bridges the gap. We keep the superior RBF performance/Recall while adding a layer of transparency to satisfy the doctors.
""")
