"""
Problem Statement:
Distance Metrics and Feature Scaling in k-NN.
Task: Analyze the impact of feature scaling on varying ranges (Age vs Cholesterol) in a k-NN diagnosis system.

Context:
- Features: Age (18-90), BP (80-200), Cholesterol (100-300), Blood Sugar (70-200).
- Patients:
  - P1: [25, 120, 180, 90]
  - P2: [30, 125, 185, 95]

Sub-Problems:
1. Part A: Why raw Euclidean distance fails. Distance contribution calculation.
2. Part B: Propose Normalization (Min-Max). Calculate normalized Age.
3. Part C: Does Manhattan distance solve scaling?

Steps to Solve:
1. Define Patient Vectors P1 and P2.
2. Calculate Raw Squared Differences for each feature.
3. Illustrate the dominance of large-scale features (Cholesterol) vs small-scale (Age).
4. Implement Min-Max Normalization function.
5. Apply Normalization to P1's Age.
6. Print explanations for Part A, B, and C as per word count constraints.

Expected Output:
- Raw distance contributions showing larger features dominate.
- Normalized Age value (0.0 to 1.0).
- TEXT Answers for Parts A and C clarifying the theory.
"""

"""
Task: Part A (Explanation & Calculation)
----------------------------------------
Requirement: Explain why using raw Euclidean distance without normalization is problematic.
Solution: Features with larger ranges (e.g., Cholesterol: 100-300) produce numerically larger differences than features with smaller ranges (e.g., Age: 18-90).
Why: Euclidean distance sums these squared differences. The "Distance" is mathematically dominated by Cholesterol, making Age effectively irrelevant to the algorithm. k-NN becomes a "Cholesterol Nearest Neighbor" algorithm.
Output: See console for squared_diff values showing this disparity.

Requirement: Calculate distance contribution.
Solution: (x2 - x1)^2 for each feature.
Output: [25, 25, 25, 25]. (In this specific example, differences happened to be identical (5), but usually they differ greatly).
"""

"""
Task: Part B (Normalization)
----------------------------------------
Requirement: Propose one normalization technique.
Solution: Min-Max Scaling. Formula: (x - min) / (max - min).
Why: Scales all features to a fixed range [0, 1]. This ensures every feature contributes equally to the distance calculation map.
Output: P1 Age (25) normalized -> (25 - 18) / (90 - 18) = 7 / 72 = 0.097.
"""

"""
Task: Part C (Manhattan Distance)
----------------------------------------
Requirement: Would Manhattan distance solve scaling?
Solution: No.
Why: Manhattan distance (|x2 - x1|) still sums the raw magnitudes. If Cholesterol difference is 50 and Age difference is 5, Cholesterol still contributes 10x more to the total distance. The metric (Euclidean vs Manhattan) changes the shape of the boundary, but NOT the scale bias.
Output: Text confirmation "No".
"""

# Why: NumPy is essential for vector math (arrays, square roots).
# Output: Module 'numpy' imported as 'np'.
import numpy as np

# ==========================================
# Part A: Raw Distance Calculation
# ==========================================

print("--- Part A: Raw Euclidean Distance Analysis ---")

# Define Data
# What: NumPy arrays representing the 4 features: [Age, BP, Cholesterol, Sugar]
# When: Initialization.
P1 = np.array([25, 120, 180, 90])
P2 = np.array([30, 125, 185, 95])

# Calculate squared differences (Component-wise)
# What: (P2 - P1)^2 for each feature index.
# When: Calculating Euclidean Distance components.
# Why: To show how much each feature 'adds' to the total distance sum.
# Output: Array of squared differences.
diff = P2 - P1
squared_diffs = diff ** 2

features = ["Age", "BP", "Cholesterol", "Sugar"]
print(f"Feature Diffs: {diff}")
print(f"Squared Contributions: {squared_diffs}")

# Total Distance
# What: Square root of sum of squared differences.
# Output: Scalar float.
distance = np.sqrt(np.sum(squared_diffs))
print(f"Total Euclidean Distance: {distance:.4f}")

# Explanation Comment
# Note: In THIS specific example P1 vs P2, all differences are '5', so contributions are equal (25).
# However, imagine a P3 with Cholesterol=300 (Max). The diff would be (300-180)^2 = 14400.
# A P3 with Age=90 (Max). The diff would be (90-25)^2 = 4225.
# Cholesterol clearly has potential to be 3x more influential than Age just by range.

# ==========================================
# Part B: Normalization (Min-Max)
# ==========================================

print("\n--- Part B: Normalization (Min-Max) ---")

# Define Ranges
# What: Min and Max values for Age as given in problem. (18 to 90).
# Why: Required for the Min-Max formula: (x - min) / (max - min).
age_min = 18
age_max = 90

# Calculation
# What: Normalize P1's Age (25).
# Output: Float between 0 and 1.
p1_age_raw = 25
p1_age_norm = (p1_age_raw - age_min) / (age_max - age_min)

print(f"P1 Raw Age: {p1_age_raw}")
print(f"Age Range: {age_min} - {age_max}")
print(f"Calculation: (25 - 18) / (90 - 18) = 7 / 72")
print(f"Normalized Age: {p1_age_norm:.4f}")

# ==========================================
# Part C: Manhattan Distance
# ==========================================

print("\n--- Part C: Manhattan Distance Check ---")

# Manhattan Calculation
# What: Sum of Absolute Differences (|x2 - x1|).
# Output: Scalar 20.
manhattan_dist = np.sum(np.abs(P2 - P1))

print(f"Manhattan Distance: {manhattan_dist}")
print("Conclusion: Manhattan distance is simply 5 + 5 + 5 + 5 = 20.")
print("Does it solve scaling? NO.")
print("Reason: If P2 had Cholesterol 300, difference is |300-180| = 120.")
print("If P2 had Age 90, difference is |90-25| = 65.")
print("Cholesterol (120) still dominates Age (65) in the total sum.")
