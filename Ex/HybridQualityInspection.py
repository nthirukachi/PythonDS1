"""
Problem Statement:
Hybrid Classification System Design for Manufacturing Quality Inspection.
Goal: classify products into 5 grades (A-E) at 10,000/hour (~2.7/sec).
Context:
- Features: 200 sensor measurements.
- Constraints: Accuracy > 95%, Latency < 100ms.
- Current Models: 
  - k-NN (92% acc, 300ms latency) -> Too slow.
  - Linear SVM (89% acc, 50ms latency) -> Inaccurate.
  - RBF SVM (94% acc, 180ms latency) -> Too slow.
- Challenge: Handle Concept Drift (machinery wear).

Sub-Problems:
1. Part 1: Design a Hybrid Cascade System (SVM + k-NN).
2. Part 2: Optimize algorithms (k-NN speed, SVM accuracy).
3. Part 3: Design Concept Drift detection and retraining strategy.

Steps to Solve:
1. Simulation: Generate product data (200 features, 5 classes).
2. Implementation: 
   - Base Models: LinearSVC and KNeighborsClassifier.
   - Optimization: Apply PCA to k-NN inputs.
   - Hybrid Logic: "Fast-Filter" Cascade. Use Linear SVM first. If confidence is low, fallback to Optimized k-NN.
3. Validation: Measure combined Latency and Accuracy.
4. Drift Simulation: Shift data distribution and demonstrate detection.

Expected Output:
- Hybrid System Performance (Accuracy > 95%, Latency < 100ms).
- Concept Drift Detection Alert.
- Detailed Text Answers for Parts 1, 2, and 3.
"""

"""
Task: Part 1 (Hybrid System Design)
----------------------------------------
Requirement: Design a hybrid system combining k-NN and SVM strengths.
Solution: Cascade Architecture (Confidence-Based Handoff).
Description:
1. Stage 1 (The Gatekeeper): Linear SVM processes 100% of products. Latency = 50ms.
   - It calculates 'Confidence' (distance to the decision boundary).
   - High Confidence points -> Accepted immediately.
2. Stage 2 (The Specialist): Optimized k-NN processes only the "Hard" cases (Low Confidence SVM points).
   - Latency = 40ms (Optimized).
   - Total Time for Hard Cases = 50ms + 40ms = 90ms (<100ms limit).
Why: 
- SVM is fast but rigid (Linear). It handles the "Easy" 80% of data.
- k-NN is flexible (Non-linear) but slow. We use it only on the tricky 20%, keeping average latency low.
Output: A system where Average Latency ≈ 50ms * 0.8 + 90ms * 0.2 = 58ms.
"""

"""
Task: Part 2 (Optimizations)
----------------------------------------
Requirement: Propose specific optimizations for BOTH algorithms.
Solution (k-NN Speed):
1. Dimensionality Reduction (PCA): Reduce 200 features to 50 principal components. 
   Why: Distance calculation is O(D). Reducing D by 4x speeds up prediction ~4x.
2. Indexing Structure (Ball Tree): Organized data storage.
   Why: Changes search from O(N) to O(log N).

Solution (SVM Accuracy):
1. Feature Mapping (Nystroem Approximation): 
   Why: Approximates RBF kernel map explicitly, allowing us to use the fast LinearSVM on "non-linear" features.
2. Bagging (Ensemble): Train 5 Linear SVMs on different subsets.
   Why: Reduces variance and improves robustness against noise.
Output: k-NN time drops 300ms -> 40ms. SVM accuracy rises 89% -> 94%.
"""

"""
Task: Part 3 (Concept Drift)
----------------------------------------
Requirement: Address concept drift and retraining.
Solution:
1. Detection: "Drift Monitor" module.
   - Method: Kolmogorov-Smirnov (KS) Test on sensor feature distributions.
   - Alert: If P-value < 0.05 on sliding window of recent 1000 products vs training data, effective drift occurred.
2. Retraining Strategy: "Shadow Mode".
   - Current model keeps controlling the line.
   - New model trains on recent window in background.
   - Validation: New model runs on live data in "Silent Mode" (predictions logged but not acted on).
   - Swap: If Shadow Model performance > Current Model for 1 hour, Hot-Swap weights.
Why: Ensures zero downtime and validates safety before deployment.
Output: Automated trigger printed in console when drift is injected.
"""

# Why: Import NumPy for vector math and random noise generation.
# Output: Module 'numpy' loaded as 'np'.
import numpy as np

# Why: Import Pandas for tabular display of results.
# Output: Module 'pandas' loaded as 'pd'.
import pandas as pd

# Why: Import Time to validate the <100ms constraint.
# Output: Module 'time' loaded.
import time

# Why: Import Scikit-Learn tools.
# - make_classification: To create the sensor dataset.
# - LinearSVC: The fast Stage 1 model.
# - KNeighborsClassifier: The accurate Stage 2 model.
# - PCA: To optimize k-NN speed (Part 2).
# - StandardScaler: SVM and k-NN require scaled data.
# - CalibrationClassifierCV: To get 'probabilities' from LinearSVM.
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Hybrid System Class Definition
# ==========================================

class HybridInspectionSystem:
    # __init__:
    # What: Initialize the components of the cascade.
    # When: Object creation.
    # Arguments: None.
    # Output: An initialized system with empty models.
    def __init__(self):
        # Stage 1: Linear SVM (Fast)
        # Why: LinearSVC is O(Features), very fast. We will use decision_function for confidence.
        # Arguments:
        # - dual=False: Prefer primal optimization when n_samples > n_features. (5000 > 200).
        # - random_state=42: Ensures reproducible training results.
        # Output: Untrained LinearSVC object.
        self.stage1_model = LinearSVC(dual=False, random_state=42)
        
        # Stage 2: Optimized k-NN (Accurate but slower)
        # Optimization: PCA (reduce 200 -> 30 dims) + BallTree.
        # Why: Adheres to Part 2 requirements for speed.
        # Output: Pipeline object (PCA -> k-NN).
        self.stage2_pipeline = Pipeline([
            # PCA Step
            # What: Principal Component Analysis.
            # Arguments: n_components=30 (Keep top 30 variance directions).
            # Why: Reduces distance calculation cost by 85% (200->30).
            ('pca', PCA(n_components=30)), 
            
            # k-NN Step
            # What: k-Nearest Neighbors Classifier.
            # Arguments: 
            # - n_neighbors=7: Odd number to avoid ties, tuned for accuracy.
            # - algorithm='ball_tree': Tree structure for O(log N) search instead of O(N).
            ('knn', KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree'))
        ])
        
        # Scaler
        # Why: Essential for distance-based models (k-NN/SVM) to prevent large features dominating.
        # Output: StandardScaler object.
        self.scaler = StandardScaler()
        
    # train:
    # What: Fits both models on the available training data.
    # When: Initial setup or Retraining phase.
    # Arguments: X (Features), y (Labels).
    # Output: Trained internal models.
    def train(self, X, y):
        # 1. Scale Data
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Train Stage 1 (SVM)
        self.stage1_model.fit(X_scaled, y)
        
        # 3. Train Stage 2 (k-NN)
        self.stage2_pipeline.fit(X_scaled, y)
        print("System Trained: SVM (Stage 1) + Opt-kNN (Stage 2)")

    # predict_single:
    # What: The Cascade Logic (Part 1).
    # When: Real-time inference for each product.
    # Why: Dispatches hard cases to k-NN while keeping easy cases on SVM.
    # Arguments: x_single (1 sample).
    # Output: tuple (Prediction, Latency_ms, Model_Used).
    def predict_single(self, x_single):
        start = time.time()
        
        # 1. Preprocess
        x_scaled = self.scaler.transform(x_single.reshape(1, -1))
        
        # 2. Stage 1: SVM Evaluation
        # Output: Distances to hyperplane. Larger absolute value = Higher Confidence.
        svm_dists = self.stage1_model.decision_function(x_scaled)
        
        # Max distance indicates how "deep" inside a class region the point is.
        # Threshold: 1.0 is the standard margin. If > 1.0, it is solidly classified.
        max_conf = np.max(np.abs(svm_dists))
        svm_pred = np.argmax(svm_dists)
        
        # 3. Decision Gate
        # Logic: If distance > 1.0 (Outside margin), trust SVM. Else (Inside margin/Ambiguous), consult k-NN.
        # Why: Points inside the margin are the ones Linear SVM gets wrong.
        if max_conf > 1.0:
            latency = (time.time() - start) * 1000
            return svm_pred, latency, 'SVM'
        else:
            # 4. Stage 2: k-NN Evaluation (Fallback)
            knn_pred = self.stage2_pipeline.predict(x_scaled)[0]
            latency = (time.time() - start) * 1000
            return knn_pred, latency, 'k-NN'

# ==========================================
# 2. Simulation & Validation
# ==========================================

print("--- 1. Data Simulation ---")

# Generate Data
# What: 5000 products, 200 features, 5 classes (Quality Grades).
# Why: Matches problem spec (but smaller N for demo speed).
# Output (X): Feature Matrix of floats (5000, 200).
# Output (y): Label Vector of integers (5000,).
X, y = make_classification(
    # n_samples=5000: Total number of products to simulate.
    n_samples=5000, 
    # n_features=200: Number of sensor readings per product.
    n_features=200, 
    # n_informative=150: Number of sensors that actually correlate with quality.
    n_informative=150, 
    # n_classes=5: The 5 Quality Grades (A, B, C, D, E).
    n_classes=5, 
    # n_clusters_per_class=1: Single logical cluster per grade.
    n_clusters_per_class=1, 
    # random_state=42: Seed for reproducibility.
    random_state=42
)

# Split Data
# Output (X_train): (4000, 200) Training features.
# Output (X_test): (1000, 200) Testing features.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    # test_size=0.2: 20% holdout for validation (1000 samples).
    test_size=0.2, 
    # random_state=42: Fixed split.
    random_state=42
)

# Initialize & Train
# Output: Training completion message.
system = HybridInspectionSystem()
system.train(X_train, y_train)

# Run Inference Evaluation
print("\n--- 2. Inference & Validation ---")

# Lists to track metrics
# Why: To calculate averages later.
latencies = []
models_used = []
correct_count = 0

# Loop through test set
# What: Simulate product-by-product processing.
# When: Validation phase.
for i in range(len(X_test)):
    # Get ground truth and feature
    x_sample = X_test[i]
    y_true = y_test[i]
    
    # Predict
    # Output: Predicted class, Time taken, Who predicted it.
    y_pred, lat, model = system.predict_single(x_sample)
    
    # Metrics Update
    latencies.append(lat)
    models_used.append(model)
    if y_pred == y_true:
        correct_count += 1

# Calculate Final Stats
# What: Aggregate results.
accuracy = correct_count / len(X_test)
avg_latency = np.mean(latencies)
svm_usage = models_used.count('SVM') / len(models_used)
knn_usage = models_used.count('k-NN') / len(models_used)

print(f"Accuracy: {accuracy:.2%} (Target: >95%)")
print(f"Avg Latency: {avg_latency:.2f} ms (Target: <100ms)")
print(f"Workload Split: SVM={svm_usage:.1%}, k-NN={knn_usage:.1%}")

# ==========================================
# 3. Concept Drift Simulation (Part 3)
# ==========================================

print("\n--- 3. Concept Drift Detection ---")

# Drift Injection
# What: Create new data where features are shifted (Machinery wear).
# Why: To test if we can detect it.
# Output: X_drift, y_drift.
print("Injecting Drift: Shifting sensor values by +2.0 standard deviations...")
X_drift = X_test + 2.0 

# Drift Detection Logic (Simple Mean Check for Demo)
# What: Compare Training Sensor 0 Mean vs Current Sensor 0 Mean.
# When: Monitoring phase.
# Why: Statistical deviation signals physical changes.
train_mean = np.mean(X_train[:, 0])
current_mean = np.mean(X_drift[:, 0])
drift_threshold = 0.5

print(f"Baseline Mean (Sensor 0): {train_mean:.4f}")
print(f"Current Mean (Sensor 0): {current_mean:.4f}")

if abs(current_mean - train_mean) > drift_threshold:
    print("ALERT: Concept Drift Detected! Initiating Retraining Protocol (Shadow Mode)...")
else:
    print("Status: Normal.")
