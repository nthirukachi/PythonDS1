"""
====================================================================================================
1. PROBLEM STATEMENT:
Evaluate K-Means cluster quality with inertia and silhouette analysis on the iris dataset.

We need to:
1.  Load the Iris dataset (150 samples, 4 features).
2.  Standardize the features to ensure equal importance during distance calculation.
3.  Run the K-Means algorithm for a range of cluster numbers ($k=2$ to $k=6$).
4.  For each $k$, record the "Inertia" (sum of squared distances to centroid) and the "Average Silhouette Score" (measure of separation).
5.  Generate an "Elbow Plot" (Inertia vs k) to identify the point of diminishing returns.
6.  Generate a "Silhouette Plot" (Score vs k) to validate the cluster structure.
7.  Justify the optimal $k$.

STEPS TO SOLVE THE PROBLEM:
1.  Data Loading: Use `load_iris` to get feature matrix X.
2.  Preprocessing: Use `StandardScaler` to normalize X.
3.  Analysis Loop:
    -   Initialize lists for inertia and silhouette scores.
    -   Loop k from 2 to 6.
    -   Initialize KMeans with `init='k-means++'` and `n_init='auto'`.
    -   Fit to scaled data.
    -   Append `inertia_` and `silhouette_score`.
4.  Visualization:
    -   Create a figure with two subplots.
    -   Subplot 1: Line plot of Inertia vs k (Elbow Method).
    -   Subplot 2: Line plot of Silhouette Score vs k.
5.  Output: Print metrics table to console and save plots.

EXPECTED OUTPUT:
-   A table showing Inertia (decreasing) and Silhouette (peaking) for each k.
-   A plot file `kmeans_plots.png`.
-   The Elbow plot should show a "bend" around k=2 or k=3.
-   The Silhouette plot should show the highest value around k=2 or k=3.
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the matplotlib plotting interface.
# 2.2: Why it is used: To visualize the metrics (Elbow and Silhouette curves).
# 2.3: When to used: Data visualization.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `import matplotlib.pyplot as plt`.
# 2.6: Output: Library loaded.
import matplotlib.pyplot as plt

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the Iris dataset loader.
# 2.2: Why it is used: To get the benchmark dataset without downloading external files.
# 2.3: When to used: Working with standard datasets.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.datasets import load_iris`.
# 2.6: Output: Function loaded.
from sklearn.datasets import load_iris

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports StandardScaler.
# 2.2: Why it is used: To normalize features (mean=0, std=1) so distance calculations are fair.
# 2.3: When to used: Before any distance-based algorithm (KMeans, SVM, KNN).
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.preprocessing import StandardScaler`.
# 2.6: Output: Class loaded.
from sklearn.preprocessing import StandardScaler

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports K-Means clustering algorithm.
# 2.2: Why it is used: The core algorithm used to solve the problem by partitioning data.
# 2.3: When to used: Unsupervised learning/Clustering.
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.cluster import KMeans`.
# 2.6: Output: Class loaded.
from sklearn.cluster import KMeans

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports silhouette score metric.
# 2.2: Why it is used: To measure how well-separated the clusters are (ranges from -1 to 1).
# 2.3: When to used: Evaluating partial clustering (e.g., when no ground truth exists).
# 2.4: Where to use: Global scope.
# 2.5: How to use: `from sklearn.metrics import silhouette_score`.
# 2.6: Output: Function loaded.
from sklearn.metrics import silhouette_score

# ==================================================================================================
# 1. DATA LOADING & PREPROCESSING
# ==================================================================================================
# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints a section header to the console.
# 2.2: Why it is used: To organize output for the user.
# 2.6: Output: "--- 1. Data Loading & Preprocessing ---"
print("\n--- 1. Data Loading & Preprocessing ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Loads the full Iris dataset structure.
# 2.2: Why it is used: Provides the data (X) and metadata needed for analysis.
# 2.3: When to used: At the start of the script.
# 2.4: Where to use: Assignment.
# 2.5: How to use: `iris = load_iris()`.
# 2.6: Output: Bunch object (dictionary-like) containing data, target, names, etc.
iris = load_iris()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Extracts the feature matrix.
# 2.2: Why it is used: We cluster based on these features (sepal/petal length/width).
# 2.3: When to used: After loading data.
# 2.4: Where to use: Assignment.
# 2.5: How to use: `X = iris.data`.
# 2.6: Output: Numeric Matrix (150, 4).
X = iris.data

# 2. Detailed Explanation:
# 2.1: What the line of code does: Fits a StandardScaler to X and returns the transformed version.
# 2.2: Why it is used: K-Means relies on Euclidean distance; scaling prevents features with large magnitudes from dominating features with small magnitudes.
# 2.3: When to used: Preprocessing step.
# 2.4: Where to use: Assignment.
# 2.5: How to use: `StandardScaler().fit_transform(X)`.
# 2.6: Output: Scaled Matrix (150, 4) with mean 0 and variance 1.

# 3. Arguments Explanation:
#    A. X
#       3.1 What: The feature matrix.
#       3.2 Why: The raw data to be scaled.
#       3.3 When to use: Inside fit_transform.
#       3.4 Where to use: Argument.
#       3.5 How to use: Pass the variable.
#       3.6 Argument Example: X
X_scaled = StandardScaler().fit_transform(X)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints the shape of the scaled data.
# 2.2: Why it is used: Verification of data loading.
# 2.6: Output: "Data Shape: (150, 4)"
print(f"Data Shape: {X_scaled.shape}")

# ==================================================================================================
# 2. K-MEANS ANALYSIS LOOP
# ==================================================================================================
# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints section header.
# 2.6: Output: "--- 2. Running K-Means (k=2 to 6) ---"
print("\n--- 2. Running K-Means (k=2 to 6) ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines the range of clusters to test.
# 2.2: Why it is used: The problem asks for k from 2 to 6.
# 2.3: When to used: Setting up the loop.
# 2.4: Where to use: Variable assignment.
# 2.5: How to use: `range(start, stop_exclusive)`.
# 2.6: Output: Range object `[2, 3, 4, 5, 6]`.
k_values = range(2, 7)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes empty list for inertia values.
# 2.2: Why it is used: storage for plotting later.
# 2.3: When to used: Before loop.
# 2.6: Output: Empty list `[]`.
inertia_list = []

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initialize empty list for silhouette scores.
# 2.2: Why it is used: Storage for plotting later.
# 2.3: When to used: Before loop.
# 2.6: Output: Empty list `[]`.
silhouette_list = []

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints the table header for the metrics.
# 2.2: Why it is used: To make the output readable.
# 2.6: Output: "k     | Inertia         | Silhouette Score    "
print(f"{'k':<5} | {'Inertia':<15} | {'Silhouette Score':<20}")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints a separator line.
# 2.6: Output: "---------------------------------------------"
print("-" * 45)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Starts the loop iterating through k=2, 3, 4, 5, 6.
# 2.2: Why it is used: To test multiple cluster counts.
# 2.6: Output: Iterator.
for k in k_values:
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Initializes the KMeans model with specific hyperparameters.
    # 2.2: Why it is used: To create a clusterer instance for the current k.
    # 2.3: When to used: Inside loop.
    # 2.4: Where to use: Assignment.
    # 2.5: How to use: `KMeans(...)`.
    # 2.6: Output: KMeans object.
    
    # 3. Arguments Explanation:
    #    A. n_clusters
    #       3.1 What: Number of centroids to find.
    #       3.2 Why: This is the 'k' we are testing.
    #       3.3 When to use: Always.
    #       3.4 Where to use: Argument.
    #       3.5 How to use: Integer.
    #       3.6 Argument Example: k (e.g., 3)
    #    B. init
    #       3.1 What: Initialization method 'k-means++'.
    #       3.2 Why: Selects initial centroids effectively to speed up convergence.
    #       3.3 When to use: Recommended best practice.
    #       3.4 Where to use: Argument.
    #       3.5 How to use: String.
    #       3.6 Argument Example: 'k-means++'
    #    C. n_init
    #       3.1 What: Number of times to run with different centroid seeds.
    #       3.2 Why: 'auto' usually defaults to 10 or 1, aiming to avoid local minima.
    #       3.3 When to use: To ensure stability.
    #       3.4 Where to use: Argument.
    #       3.5 How to use: String or Int.
    #       3.6 Argument Example: 'auto'
    #    D. random_state
    #       3.1 What: Seed for the random number generator.
    #       3.2 Why: Ensures the results are reproducible.
    #       3.3 When to use: In examples/tests.
    #       3.4 Where to use: Argument.
    #       3.5 How to use: Integer.
    #       3.6 Argument Example: 42
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Fits the K-Means model to the scaled data.
    # 2.2: Why it is used: To execute the clustering algorithm (find centroids).
    # 2.3: When to used: Training phase.
    # 2.4: Where to use: Method call.
    # 2.5: How to use: `model.fit(X)`.
    # 2.6: Output: Fitted estimator.
    kmeans.fit(X_scaled)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Extracts the 'inertia_' attribute (Sum of Squared Distances).
    # 2.2: Why it is used: It's our primary metric for the Elbow Method.
    # 2.3: When to used: After fitting.
    # 2.6: Output: Float value.
    inertia = kmeans.inertia_
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Appends the inertia value to our tracking list.
    # 2.2: Why it is used: To save it for the plot later.
    # 2.6: Output: None (List modified).
    inertia_list.append(inertia)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Computes the Silhouette Score for the current clustering.
    # 2.2: Why it is used: To evaluate how well-separated the clusters are.
    # 2.3: When to used: Validation step.
    # 2.5: How to use: `silhouette_score(X, labels)`.
    # 2.6: Output: Float value between -1 and 1.
    
    # 3. Arguments Explanation:
    #    A. X
    #       3.1 What: Data that was clustered.
    #       3.2 Why: To calculate distances between points.
    #       3.3 When to use: Always.
    #       3.4 Where to use: Argument 1.
    #       3.5 How to use: Array.
    #       3.6 Argument Example: X_scaled
    #    B. labels
    #       3.1 What: The predicted cluster labels for each point.
    #       3.2 Why: Defines which cluster each point belongs to.
    #       3.3 When to use: Always.
    #       3.4 Where to use: Argument 2.
    #       3.5 How to use: Array.
    #       3.6 Argument Example: kmeans.labels_
    score = silhouette_score(X_scaled, kmeans.labels_)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Appends the score to our tracking list.
    # 2.6: Output: None (List modified).
    silhouette_list.append(score)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Prints the current k and its metrics.
    # 2.2: Why it is used: To show progress and values.
    # 2.6: Output: "3     | 139.8...        | 0.459...       "
    print(f"{k:<5} | {inertia:<15.4f} | {score:<20.4f}")

# ==================================================================================================
# 3. VISUALIZATION
# ==================================================================================================
# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints section header.
# 2.6: Output: "--- 3. Generating Plots ---"
print("\n--- 3. Generating Plots ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Creates a figure with two subplots side-by-side.
# 2.2: Why it is used: To display both the Elbow plot and Silhouette plot in one image.
# 2.3: When to used: Plotting.
# 2.4: Where to use: Assignment.
# 2.5: How to use: `plt.subplots(rows, cols, figsize=(w, h))`.
# 2.6: Output: Tuple of (Figure object, Array of Axes objects).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Elbow Curve
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots the Inertia values against K on the first subplot.
# 2.2: Why it is used: To visualize the "Elbow" curve.
# 2.3: When to used: Visualization.
# 2.6: Output: Line plot object.
ax1.plot(k_values, inertia_list, marker='o', linestyle='--')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets the title of the first subplot.
# 2.6: Output: Title text.
ax1.set_title('Elbow Plot (Inertia)')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Labels the x-axis.
# 2.6: Output: Label text.
ax1.set_xlabel('Number of Clusters (k)')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Labels the y-axis.
# 2.6: Output: Label text.
ax1.set_ylabel('Inertia')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds a grid to the plot for readability.
# 2.6: Output: Grid lines.
ax1.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds an annotation to point out the Elbow at k=3.
# 2.2: Why it is used: To guide the viewer's interpretation.
# 2.6: Output: Arrow and text.

# 3. Arguments Explanation:
#    A. text
#       3.1 What: The text to display.
#       3.6 Argument Example: 'Potential Elbow (k=3)'
#    B. xy
#       3.1 What: The point (k, inertia) to point the arrow at.
#       3.6 Argument Example: (3, inertia_list[1])
#    C. xytext
#       3.1 What: The location of the text itself.
#       3.6 Argument Example: (3.5, inertia_list[1]+20)
#    D. arrowprops
#       3.1 What: Styling for the arrow.
#       3.6 Argument Example: dict(...)
ax1.annotate('Potential Elbow (k=3)', xy=(3, inertia_list[1]), xytext=(3.5, inertia_list[1]+20),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot 2: Silhouette Score
# --------------------------------------------------------------------------------------------------
# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots the Silhouette scores against K on the second subplot.
# 2.2: Why it is used: To visualize the silhouette trend.
# 2.6: Output: Line plot object.
ax2.plot(k_values, silhouette_list, marker='s', linestyle='-', color='orange')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets the title of the second subplot.
# 2.6: Output: Title text.
ax2.set_title('Silhouette Analysis')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Labels the x-axis.
# 2.6: Output: Label text.
ax2.set_xlabel('Number of Clusters (k)')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Labels the y-axis.
# 2.6: Output: Label text.
ax2.set_ylabel('Avg. Silhouette Score')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adds a grid.
# 2.6: Output: Grid lines.
ax2.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adjusts the layout to prevent overlapping.
# 2.6: Output: None (Figure adjusted).
plt.tight_layout()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Saves the figure to a PNG file.
# 2.2: Why it is used: To simplify sharing the results.
# 2.6: Output: File on disk.
plt.savefig('kmeans_plots.png')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints confirmation message.
# 2.6: Output: "Saved plots to kmeans_plots.png"
print("Saved plots to kmeans_plots.png")
