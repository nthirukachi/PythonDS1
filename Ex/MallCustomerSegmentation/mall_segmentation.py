"""
====================================================================================================
1. PROBLEM STATEMENT:
Evaluate Silhouette Score on Real Data [CODING] - Mall Customer Segmentation.

We need to:
1.  Generate a synthetic dataset that mimics 'Mall_Customers.csv' (Annual Income vs Spending Score).
2.  Preprocess the data (StandardScaler).
3.  Run K-Means for K = {2, 3, 4, 5}.
4.  Compute and tabulate "Average Silhouette Score" and "Inertia" for each K.
5.  Generate "Silhouette Diagrams" (knife plots) for the best-performing K (likely 5).
6.  Produce an Elbow Plot.

STEPS TO SOLVE THE PROBLEM:
1.  Data Generation: Use `make_blobs` to create 5 distinct clusters corresponding to standard customer segments (Low-Low, Low-High, Mid-Mid, High-Low, High-High).
2.  Preprocessing: Scale features using `StandardScaler`.
3.  Analysis Loop:
    -   Iterate K from 2 to 5.
    -   Fit KMeans.
    -   Calculate Inertia and Silhouette Score.
4.  Visualization:
    -   Plot Inertia (Elbow) and Avg Silhouette Score.
    -   For Optimal K (5), generate the detailed Silhouette Coefficient plot for every sample.
5.  Output: Save plots to 'mall_analysis.png' and 'silhouette_k5.png'.

EXPECTED OUTPUT:
-   A table showing Silhouette Score peaks at K=5.
-   Elbow plot showing a bend at K=5.
-   Silhouette plot showing 5 relatively even "knives" (clusters).
====================================================================================================
"""

# ==================================================================================================
# IMPORT LIBRARIES
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the numpy library and aliases it as `np`.
# 2.2: Why it is used: To perform efficient numerical operations on arrays, which is required for data manipulation.
# 2.3: When to used: Whenever we need to work with matrices, vectors, or mathematical functions.
# 2.4: Where to use: At the beginning of the script (global scope).
# 2.5: How to use: `import numpy as np`
# 2.6: How it works: Loads the compiled C-optimized NumPy library into memory.
# 2.7: Output: The module object `np` is available for use.
import numpy as np

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the pyplot module from matplotlib and aliases it as `plt`.
# 2.2: Why it is used: To generate visualizations like scatter plots, line charts, and silhouette diagrams.
# 2.3: When to used: When we need to create visual representations of data.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `import matplotlib.pyplot as plt`
# 2.6: How it works: Provides a state-machine interface to the underlying plotting backend.
# 2.7: Output: The module object `plt` is available.
import matplotlib.pyplot as plt

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the colormap module from matplotlib as `cm`.
# 2.2: Why it is used: To generate distinct colors for different clusters in the silhouette plot.
# 2.3: When to used: When we need to map numerical values (cluster IDs) to colors.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `import matplotlib.cm as cm`
# 2.6: How it works: Provides access to standard colormaps (like 'nipy_spectral').
# 2.7: Output: The module object `cm` is available.
import matplotlib.cm as cm

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the `make_blobs` function from sklearn.datasets.
# 2.2: Why it is used: To generate synthetic clustering data that mimics the Mall dataset structure.
# 2.3: When to used: When we need controlled, reproducible data for testing algorithms.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `from sklearn.datasets import make_blobs`
# 2.6: How it works: Generates isotropic Gaussian blobs for clustering.
# 2.7: Output: The function `make_blobs` is available.
from sklearn.datasets import make_blobs

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the `KMeans` class from sklearn.cluster.
# 2.2: Why it is used: To perform the K-Means clustering algorithm.
# 2.3: When to used: When we want to partition data into K distinct non-overlapping subgroups.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `from sklearn.cluster import KMeans`
# 2.6: How it works: Loads the class definition for K-Means.
# 2.7: Output: The class `KMeans` is available.
from sklearn.cluster import KMeans

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports silhouette validation metrics from sklearn.metrics.
# 2.2: Why it is used: To evaluate the quality of the clusters (how similar an object is to its own cluster vs other clusters).
# 2.3: When to used: To determine the optimal number of clusters or validate consistency.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `from sklearn.metrics import silhouette_samples, silhouette_score`
# 2.6: How it works: Loads the functions `silhouette_samples` and `silhouette_score`.
# 2.7: Output: The functions are available.
from sklearn.metrics import silhouette_samples, silhouette_score

# 2. Detailed Explanation:
# 2.1: What the line of code does: Imports the `StandardScaler` class from sklearn.preprocessing.
# 2.2: Why it is used: To normalize features by removing the mean and scaling to unit variance.
# 2.3: When to used: Before feeding data into distance-based algorithms like K-Means.
# 2.4: Where to use: At the beginning of the script.
# 2.5: How to use: `from sklearn.preprocessing import StandardScaler`
# 2.6: How it works: Loads the scaler class.
# 2.7: Output: The class `StandardScaler` is available.
from sklearn.preprocessing import StandardScaler

# ==================================================================================================
# 1. DATA GENERATION & PREPROCESSING
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints a header message to the console.
# 2.2: Why it is used: To inform the user that the data generation step is starting.
# 2.3: When to used: For logging execution progress.
# 2.4: Where to use: Before the data generation block.
# 2.5: How to use: `print("message")`
# 2.6: How it works: Writes string to standard output.
# 2.7: Output: Text '--- 1. Generating Synthetic Mall Data ---' in console.
print("\n--- 1. Generating Synthetic Mall Data ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines the centroids for the 5 clusters representing customer segments.
# 2.2: Why it is used: To explicitly structure the synthetic data to look like 'Mall Customers' (Income vs Score).
# 2.3: When to used: When customizing `make_blobs` centers.
# 2.4: Where to use: Before calling `make_blobs`.
# 2.5: How to use: List of [x, y] coordinates.
# 2.6: How it works: Creates a list of lists.
# 2.7: Output: A list variable `centers` containing 5 coordinate pairs.
centers = [
    [25, 20],  # Low Income, Low Score
    [25, 80],  # Low Income, High Score
    [60, 50],  # Mid Income, Mid Score
    [90, 20],  # High Income, Low Score
    [90, 80]   # High Income, High Score
]

# 2. Detailed Explanation:
# 2.1: What the line of code does: Generates the synthetic dataset (features X and labels y_true).
# 2.2: Why it is used: To create the raw data for analysis.
# 2.3: When to used: To initialize the dataset.
# 2.4: Where to use: After defining centers.
# 2.5: How to use: Call `make_blobs` with parameters.
# 2.6: How it works: Generates random points around the specified centers using a Gaussian distribution.
# 2.7: Output: Tuple `(X, y_true)` containing feature array and cluster labels.
# 3. Arguments:
#    A. n_samples
#       3.1 What: The total number of points to generate.
#       3.2 Why: Defines dataset size.
#       3.3 When: Always required.
#       3.4 Where: Function argument.
#       3.5 How: `n_samples=200`
#       3.6 Example: 200
#    B. centers
#       3.1 What: The center coordinates of clusters.
#       3.2 Why: To define the "segments".
#       3.3 When: Customizing topology.
#       3.4 Where: Function argument.
#       3.5 How: `centers=centers`
#       3.6 Example: [[25, 20], [25, 80]...]
#    C. cluster_std
#       3.1 What: The standard deviation of the clusters.
#       3.2 Why: To control how "spread out" the points are.
#       3.3 When: Tuning difficulty.
#       3.4 Where: Function argument.
#       3.5 How: `cluster_std=5.0`
#       3.6 Example: 5.0
#    D. random_state
#       3.1 What: Seed for random number generator.
#       3.2 Why: To ensure reproducibility.
#       3.3 When: Testing.
#       3.4 Where: Function argument.
#       3.5 How: `random_state=42`
#       3.6 Example: 42
X, y_true = make_blobs(n_samples=200, centers=centers, cluster_std=5.0, random_state=42)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Scales the features to have mean 0 and variance 1.
# 2.2: Why it is used: K-Means is sensitive to scale; Income (0-100k) would dominate Score (0-100) without this.
# 2.3: When to used: Preprocessing step.
# 2.4: Where to use: Before clustering.
# 2.5: How to use: `StandardScaler().fit_transform(X)`
# 2.6: How it works: Calculates mean/std of X, then applies z = (x-mean)/std.
# 2.7: Output: `X_scaled`, a numpy array of transformed features.
X_scaled = StandardScaler().fit_transform(X)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints the shape of the generated dataset.
# 2.2: Why it is used: Verification of data dimensions.
# 2.3: When to used: Debugging/Logging.
# 2.4: Where to use: After generation.
# 2.5: How to use: `print(...)`
# 2.6: How it works: String formatting + print.
# 2.7: Output: Text like "Dataset Generated: (200, 2)".
print(f"Dataset Generated: {X_scaled.shape}")

# ==================================================================================================
# 2. ANALYSIS LOOP (K=2..5)
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints header for Analysis section.
# 2.2: Why it is used: User feedback.
# 2.3: When to used: Logging.
# 2.4: Where to use: Before loop.
# 2.5: How to use: `print(...)`
# 2.6: How it works: Writes to stdout.
# 2.7: Output: Text '--- 2. Running K-Means Analysis ---'
print("\n--- 2. Running K-Means Analysis ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Defines the list of K values to test.
# 2.2: Why it is used: To iterate through different cluster counts.
# 2.3: When to used: Loop setup.
# 2.4: Where to use: Before for-loop.
# 2.5: How to use: `[2, 3, 4, 5]`
# 2.6: How it works: Creates a list.
# 2.7: Output: List `range_n_clusters`.
range_n_clusters = [2, 3, 4, 5]

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes an empty list for inertia scores.
# 2.2: Why it is used: To store results for the Elbow Plot.
# 2.3: When to used: Metric tracking.
# 2.4: Where to use: Before loop.
# 2.5: How to use: `[]`
# 2.6: How it works: Creates empty list.
# 2.7: Output: List `inertia_list`.
inertia_list = []

# 2. Detailed Explanation:
# 2.1: What the line of code does: Initializes an empty list for silhouette scores.
# 2.2: Why it is used: To store results for the evaluation plot.
# 2.3: When to used: Metric tracking.
# 2.4: Where to use: Before loop.
# 2.5: How to use: `[]`
# 2.6: How it works: Creates empty list.
# 2.7: Output: List `silhouette_avg_list`.
silhouette_avg_list = []

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints the table header for metrics.
# 2.2: Why it is used: To create a readable table output.
# 2.3: When to used: Logging.
# 2.4: Where to use: Before loop.
# 2.5: How to use: f-string padding.
# 2.6: How it works: Formats text columns.
# 2.7: Output: Table header string.
print(f"{'K':<5} | {'Inertia':<15} | {'Avg Silhouette':<20}")

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints a separator line.
# 2.2: Why it is used: Visual separation.
# 2.3: When to used: Table formatting.
# 2.4: Where to use: After header.
# 2.5: How to use: String multiplication.
# 2.6: How it works: Creates line of dashes.
# 2.7: Output: '---------------------------------------------'
print("-" * 45)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Starts the loop iterating through each K value.
# 2.2: Why it is used: To test multiple hypotheses for K.
# 2.3: When to used: Analysis.
# 2.4: Where to use: Loop block.
# 2.5: How to use: `for n in list:`
# 2.6: How it works: Sets `n_clusters` to 2, then 3, etc.
# 2.7: Output: Loop context.
for n_clusters in range_n_clusters:
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Instantiates a KMeans object with the current K.
    # 2.2: Why it is used: To configure the algorithm.
    # 2.3: When to used: Inside loop.
    # 2.4: Where to use: Before fitting.
    # 2.5: How to use: `KMeans(...)`
    # 2.6: How it works: Creates an object with params.
    # 2.7: Output: Object `clusterer`.
    # 3. Arguments:
    #    A. n_clusters
    #       3.1 What: Number of clusters to find.
    #       3.2 Why: The core hyperparameter K.
    #       3.3 When: Instantiation.
    #       3.4 Where: Argument.
    #       3.5 How: `n_clusters=n_clusters`
    #       3.6 Example: 2
    #    B. n_init
    #       3.1 What: Number of re-runs with different seeds.
    #       3.2 Why: To avoid local minima.
    #       3.3 When: Instantiation.
    #       3.4 Where: Argument.
    #       3.5 How: `n_init="auto"`
    #       3.6 Example: "auto"
    #    C. random_state
    #       3.1 What: Seed.
    #       3.2 Why: Reproducibility.
    #       3.3 When: Instantiation.
    #       3.4 Where: Argument.
    #       3.5 How: `random_state=10`
    #       3.6 Example: 10
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Fits the model to data and predicts cluster labels.
    # 2.2: Why it is used: To assign each point to a cluster.
    # 2.3: When to used: Training.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: `fit_predict(X_scaled)`
    # 2.6: How it works: Runs K-Means algo.
    # 2.7: Output: Array `cluster_labels`.
    cluster_labels = clusterer.fit_predict(X_scaled)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Calculates the mean Silhouette Score for the entire dataset.
    # 2.2: Why it is used: To evaluate how well-separated the current K clustering is.
    # 2.3: When to used: Evaluation.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: `silhouette_score(X, labels)`
    # 2.6: How it works: Averages the silhouette coefficient of all samples.
    # 2.7: Output: Float `silhouette_avg`.
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Appends the calculated score to the list.
    # 2.2: Why it is used: For plotting later.
    # 2.3: When to used: Storage.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: `.append(val)`
    # 2.6: How it works: List growth.
    # 2.7: Output: Updated list.
    silhouette_avg_list.append(silhouette_avg)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Appends the inertia (sum of squared distances) to the list.
    # 2.2: Why it is used: For Elbow Plot.
    # 2.3: When to used: Storage.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: `clusterer.inertia_`
    # 2.6: How it works: Accesses attribute.
    # 2.7: Output: Updated list.
    inertia_list.append(clusterer.inertia_)
    
    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Prints the specific metrics for this K to the console.
    # 2.2: Why it is used: Immediate feedback/Table row.
    # 2.3: When to used: Logging.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: f-string formatting.
    # 2.6: How it works: Stdout.
    # 2.7: Output: Text Row (e.g. "2     | 400.00          | 0.4500").
    print(f"{n_clusters:<5} | {clusterer.inertia_:<15.2f} | {silhouette_avg:<20.4f}")

    # 2. Detailed Explanation:
    # 2.1: What the line of code does: Checks if the current K is 5.
    # 2.2: Why it is used: We only want to generate the detailed Silhouette Knife Plot for the best K (5) to save resources.
    # 2.3: When to used: Conditional execution.
    # 2.4: Where to use: Inside loop.
    # 2.5: How to use: `if ... == 5:`
    # 2.6: How it works: Comparison.
    # 2.7: Output: Boolean flow control.
    if n_clusters == 5:
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Creates a new figure and axes for the plot.
        # 2.2: Why it is used: Container for the plot.
        # 2.3: When to used: Plotting.
        # 2.4: Where to use: Inside if-block.
        # 2.5: How to use: `plt.subplots(...)`
        # 2.6: How it works: Returns Figure and Axis objects.
        # 2.7: Output: Tuple `(fig, ax1)`.
        # 3. Arguments:
        #    A. figsize
        #       3.1 What: Size of the image.
        #       3.2 Why: Readability.
        #       3.3 When: Plot creation.
        #       3.4 Where: Argument.
        #       3.5 How: `figsize=(8, 6)`
        #       3.6 Example: (8, 6)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Sets X-axis limits for the silhouette coefficients.
        # 2.2: Why it is used: Coefficients range from -1 to 1; we focus on -0.1 to 1 for clarity.
        # 2.3: When to used: Plot formatting.
        # 2.4: Where to use: Inside if-block.
        # 2.5: How to use: `.set_xlim(...)`
        # 2.6: How it works: Sets axis properties.
        # 2.7: Output: Adjusted axis.
        ax1.set_xlim([-0.1, 1])
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Computes silhouette scores for EACH sample individually.
        # 2.2: Why it is used: Required to draw the shape (knife) for each cluster.
        # 2.3: When to used: Visualization data prep.
        # 2.4: Where to use: Inside if-block.
        # 2.5: How to use: `silhouette_samples(X, labels)`
        # 2.6: How it works: Returns array of shape (n_samples,).
        # 2.7: Output: Array `sample_silhouette_values`.
        sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Initializes `y_lower` for stacking the plots vertically.
        # 2.2: Why it is used: To position the first cluster's knife at the bottom.
        # 2.3: When to used: Before cluster loop.
        # 2.4: Where to use: Variable init.
        # 2.5: How to use: `var = 10`
        # 2.6: How it works: Assignment.
        # 2.7: Output: `y_lower = 10`.
        y_lower = 10
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Loops through each cluster index (0 to K-1).
        # 2.2: Why it is used: To draw one knife shape per cluster.
        # 2.3: When to used: Plot construction.
        # 2.4: Where to use: Inner loop.
        # 2.5: How to use: `range(n_clusters)`
        # 2.6: How it works: Estimator iteration.
        # 2.7: Output: Index `i`.
        for i in range(n_clusters):
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Filters silhouette values for only the current cluster `i`.
            # 2.2: Why it is used: We need to plot this specific cluster's shape.
            # 2.3: When to used: Data slicing.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: Boolean indexing `[cluster_labels == i]`.
            # 2.6: How it works: Masking.
            # 2.7: Output: Array `ith_cluster_silhouette_values`.
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Sorts the values.
            # 2.2: Why it is used: Visual aesthetics; produces the smooth "knife" shape.
            # 2.3: When to used: Before plotting.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `.sort()`
            # 2.6: How it works: In-place sort.
            # 2.7: Output: Sorted array.
            ith_cluster_silhouette_values.sort()
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Gets the number of samples in this cluster.
            # 2.2: Why it is used: To calculate the height (`y_upper`) of this cluster's blob on the plot.
            # 2.3: When to used: Layout calculation.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `.shape[0]`
            # 2.6: How it works: Array dimension check.
            # 2.7: Output: Integer `size_cluster_i`.
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Calculates the top Y coordinate for this cluster.
            # 2.2: Why it is used: Defines the vertical span of the shape.
            # 2.3: When to used: Layout calculation.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `lower + size`
            # 2.6: How it works: Addition.
            # 2.7: Output: Integer `y_upper`.
            y_upper = y_lower + size_cluster_i
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Generates a unique color for this cluster.
            # 2.2: Why it is used: Visual distinction.
            # 2.3: When to used: Coloring.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `cm.nipy_spectral(fraction)`
            # 2.6: How it works: Maps scalar to RGBA.
            # 2.7: Output: Color tuple.
            color = cm.nipy_spectral(float(i) / n_clusters)
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Fills the area to draw the silhouette shape.
            # 2.2: Why it is used: Renders the actual plot element.
            # 2.3: When to used: Plotting.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `fill_betweenx(...)`
            # 2.6: How it works: Draws polygons.
            # 2.7: Output: Polygon on axis.
            # 3. Arguments:
            #    A. y
            #       3.1 What: Y coordinates range.
            #       3.6 Example: np.arange(10, 50)
            #    B. x1
            #       3.1 What: Left X boundary (0).
            #       3.6 Example: 0
            #    C. x2
            #       3.1 What: Right X boundary (values).
            #       3.6 Example: [0.1, 0.4, 0.5...]
            #    D. facecolor
            #       3.1 What: Fill color.
            #       3.6 Example: color tuple
            #    E. alpha
            #       3.1 What: Transparency.
            #       3.6 Example: 0.7
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Places the cluster label number in the middle of the shape.
            # 2.2: Why it is used: Identification.
            # 2.3: When to used: Labeling.
            # 2.4: Where to use: Inside inner loop.
            # 2.5: How to use: `.text(x, y, string)`
            # 2.6: How it works: Draws text.
            # 2.7: Output: Text '0', '1', etc on plot.
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # 2. Detailed Explanation:
            # 2.1: What the line of code does: Updates `y_lower` for the next cluster.
            # 2.2: Why it is used: To stack the next cluster above the current one with a small gap (10).
            # 2.3: When to used: Loop state update.
            # 2.4: Where to use: End of inner loop.
            # 2.5: How to use: `+= 10`
            # 2.6: How it works: Advance cursor.
            # 2.7: Output: Updated `y_lower`.
            y_lower = y_upper + 10
            
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Sets the main title.
        # 2.2: Why it is used: Context.
        # 2.7: Output: Title text.
        ax1.set_title("The silhouette plot for the various clusters.")
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Sets X label.
        # 2.7: Output: Label text.
        ax1.set_xlabel("The silhouette coefficient values")
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Sets Y label.
        # 2.7: Output: Label text.
        ax1.set_ylabel("Cluster label")
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Draws a vertical dashed line at the average silhouette score.
        # 2.2: Why it is used: To compare individual clusters against the global average.
        # 2.3: When to used: Annotation.
        # 2.4: Where to use: Plot finalization.
        # 2.5: How to use: `axvline(x, ...)`
        # 2.6: How it works: Draws line.
        # 2.7: Output: Red line on plot.
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Saves the figure to disk.
        # 2.2: Why it is used: To export the artifact.
        # 2.3: When to used: IO.
        # 2.4: Where to use: End of block.
        # 2.5: How to use: `savefig(filename)`
        # 2.6: How it works: Renders to file.
        # 2.7: Output: File 'silhouette_k5.png'.
        plt.savefig(f'silhouette_k{n_clusters}.png')
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Logs save confirmation.
        # 2.7: Output: Console text.
        print(f"Saved silhouette plot for K={n_clusters}")
        
        # 2. Detailed Explanation:
        # 2.1: What the line of code does: Closes the plot.
        # 2.2: Why it is used: Free memory.
        # 2.7: Output: Cleared memory.
        plt.close()

# ==================================================================================================
# 3. METRICS VISUALIZATION
# ==================================================================================================

# 2. Detailed Explanation:
# 2.1: What the line of code does: Prints header for plotting section.
# 2.7: Output: Text.
print("\n--- 3. Generating Analysis Plots ---")

# 2. Detailed Explanation:
# 2.1: What the line of code does: creates a figure with 2 subplots side-by-side.
# 2.2: Why it is used: To show Elbow and Silhouette plots together.
# 2.3: When to used: Comparison plotting.
# 2.4: Where to use: Setup.
# 2.5: How to use: `subplots(1, 2, ...)`
# 2.6: How it works: Layout engine.
# 2.7: Output: Figure and 2 Axes.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots Inertia values on the left subplot.
# 2.2: Why it is used: Elbow method visualization.
# 2.3: When to used: Plotting.
# 2.4: Where to use: Ax1.
# 2.5: How to use: `.plot(x, y, ...)`
# 2.6: How it works: Connects points.
# 2.7: Output: Line chart.
ax1.plot(range_n_clusters, inertia_list, marker='o')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets title for left plot.
# 2.7: Output: Title.
ax1.set_title('Elbow Plot (Inertia)')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets X label.
# 2.7: Output: Label.
ax1.set_xlabel('K')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets Y label.
# 2.7: Output: Label.
ax1.set_ylabel('Inertia')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Enables grid.
# 2.7: Output: Grid lines.
ax1.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Plots Silhouette Scores on right subplot.
# 2.2: Why it is used: Validation visualization.
# 2.7: Output: Line chart.
ax2.plot(range_n_clusters, silhouette_avg_list, marker='s', color='orange')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets title for right plot.
# 2.7: Output: Title.
ax2.set_title('Avg Silhouette Score')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets X label.
# 2.7: Output: Label.
ax2.set_xlabel('K')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Sets Y label.
# 2.7: Output: Label.
ax2.set_ylabel('Score')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Enables grid.
# 2.7: Output: Grid lines.
ax2.grid(True)

# 2. Detailed Explanation:
# 2.1: What the line of code does: Adjusts subplots to fit neatly.
# 2.2: Why it is used: Prevent overlap.
# 2.7: Output: Adjusted layout.
plt.tight_layout()

# 2. Detailed Explanation:
# 2.1: What the line of code does: Saves the combined analysis plot.
# 2.7: Output: File 'mall_analysis.png'.
plt.savefig('mall_analysis.png')

# 2. Detailed Explanation:
# 2.1: What the line of code does: Logs final success message.
# 2.7: Output: Console text.
print("Saved metrics summary to mall_analysis.png")
