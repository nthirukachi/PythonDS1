"""
Problem Statement:
Perform an end-to-end classification pipeline using the Social Network Ads dataset or any binary dataset.
Steps:
1. Load and preprocess the data (handle missing values, encode categorical columns).
2. Train a Logistic Regression model.
3. Plot the sigmoid curve for one feature (e.g., Age vs Purchase Probability).
4. Tune threshold values (0.3, 0.5, 0.7) and compare Precision, Recall, and F1-score.
5. Visualize the decision boundary and interpret what it means in plain language.
6. Write three key insights from your results about model performance and business impact.
"""
# Why: Import numpy for numerical operations and array handling.
# Output: Module 'numpy' loaded as 'np'.
import numpy as np

# Why: Import pandas for data manipulation and analysis (DataFrames).
# Output: Module 'pandas' loaded as 'pd'.
import pandas as pd

# Why: Import matplotlib.pyplot for creating static, animated, and interactive visualizations.
# Output: Module 'matplotlib.pyplot' loaded as 'plt'.
import matplotlib.pyplot as plt

# Why: Import seaborn for high-level interface for drawing attractive and informative statistical graphics.
# Output: Module 'seaborn' loaded as 'sns'.
import seaborn as sns

# Why: Import train_test_split to split data arrays into two subsets: for training data and for testing data.
# Output: Function 'train_test_split' imported.
from sklearn.model_selection import train_test_split

# Why: Import StandardScaler to standardize features by removing the mean and scaling to unit variance.
# Output: Class 'StandardScaler' imported.
from sklearn.preprocessing import StandardScaler

# Why: Import LogisticRegression classifier to implement the logistic regression algorithm.
# Output: Class 'LogisticRegression' imported.
from sklearn.linear_model import LogisticRegression

# Why: Import metric functions to evaluate the performance of the classification model.
# Output: Functions 'recall_score', 'precision_score', 'f1_score' imported.
from sklearn.metrics import recall_score, precision_score, f1_score

# ==========================================
# Step 1: Load and Preprocess the Data
# ==========================================

# Generates a synthetic dataset similar to 'Social Network Ads'
# Why: To ensure we have data to work with for this classification demonstration.
# logic: We create random Ages and Salaries. 
# Purchase decision is roughly based on higher age and salary plus some noise.
# Why: Seed the random number generator to ensure the following random numbers are reproducible.
# Output: Random state set to 0. Next random calls will be deterministic.
np.random.seed(0)

# Why: Define the number of samples (rows) to generate for our synthetic dataset.
# Output: Integer variable n_samples set to 400.
n_samples = 400
# np.random.randint(low, high, size):
# - 18 (low): The lowest integer to be drawn (inclusive).
# - 60 (high): The one above the largest (exclusive). Max age will be 59.
# - size=n_samples: Output shape. We want 400 random ages.
# Examples: 
#   np.random.randint(1, 7, size=10) -> simulates 10 dice rolls (1-6)
#   np.random.randint(0, 2, size=5) -> simulates 5 coin flips (0 or 1)
# Why: Generate synthetic Age data for 400 users between 18 and 59.
# Output: Array of 400 integers (e.g., [25, 42, 19, ...]).
ages = np.random.randint(18, 60, size=n_samples)
# np.random.randint(low, high, size):
# - 15000: Min salary
# - 150000: Max salary (exclusive)
# - size=n_samples: 400 samples
# Why: Generate synthetic feature Salary for 400 users between 15k and 150k.
# Output: Array of 400 integers (e.g., [20000, 85000, ...]).
salaries = np.random.randint(15000, 150000, size=n_samples)

# Create a target variable (Purchased) with some logic:
# formula: if (Age + Salary/1000) > 80 then mostly 1 else 0 (with noise)
# This mimics a linear decision boundary which Logistic Regression solves well.
# Why: Calculate a linear score based on Age and Salary to simulate a decision process.
# Output: Array of 400 floats. Higher values for older/richer people.
linear_comb = ages + salaries / 1000

# sigmoid function: 1 / (1 + e^-z). Squashes values to range [0, 1].
# shift (-90) and scale (/5) are arbitrary to control the steepness and center of the curve.
# Why: Convert the linear score into a probability between 0 and 1.
# Output: Array of 400 probabilities (e.g., 0.1, 0.9, 0.4...).
probabilities = 1 / (1 + np.exp(-(linear_comb - 90)/5)) 

# Generate labels based on probability
# np.random.rand() returns random float [0.0, 1.0).
# If random < p, we say 1 (Purchased). This adds some noise so it's not perfectly separable.
# Why: Assign binary classes (0 or 1) based on the calculated probabilities.
# Output: List of 400 binary values (0 or 1).
purchased = [1 if np.random.rand() < p else 0 for p in probabilities]

# Create DataFrame
# pd.DataFrame: creates a tabular data structure (rows and columns).
# Why: Organize the data into a structured table for easier manipulation and inspection.
# Output: A Pandas DataFrame 'df' with 3 columns: Age, EstimatedSalary, Purchased.
df = pd.DataFrame({
    'Age': ages,
    'EstimatedSalary': salaries,
    'Purchased': purchased
})

# Why: Print text to console to indicate what is being shown.
# Output: Prints "First 5 rows of the dataset:".
print("First 5 rows of the dataset:")

# Why: Display the first 5 rows to verify the data creation looked correct.
# Output: Table of first 5 rows printed to console.
print(df.head())

# Feature Matrix (X) and Target Vector (y)
# df.iloc[rows, cols]: Integer-location based indexing.
# [:, :-1] -> All rows, All columns except the last one (Features: Age, EstimatedSalary)
# .values -> Converts DataFrame to NumPy array (required for some sklearn internals, though DF often works too)
# Why: Separate the input features (Age, Salary) from the target (Purchased).
# Output: X is a 2D numpy array (400, 2).
X = df.iloc[:, :-1].values 

# [:, -1] -> All rows, Only the last column (Target: Purchased)
# Why: Isolate the target variable we want to predict.
# Output: y is a 1D numpy array (400,).
y = df.iloc[:, -1].values

# Splitting the dataset into Training and Test set
# train_test_split(*arrays, test_size, random_state):
# - *arrays: X and y (must be same length).
# - test_size=0.25: 25% of data used for testing (100 samples), 75% for training (300 samples).
#   Example: If you have 1000 rows, test_size=0.2 means 200 for test.
# - random_state=0: Seed for random number generator.
#   Why: Ensures that if you run this code again, you get the EXACT same split. Essential for reproducible results.
# Why: Create separate datasets for training and testing to evaluate model performance on unseen data.
# Output: 4 arrays: X_train (300, 2), X_test (100, 2), y_train (300,), y_test (100,).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
# StandardScaler(): Standardize features by removing the mean and scaling to unit variance.
# z = (x - u) / s
# Why?
# - Logistic Regression uses optimization solvers (like L-BFGS) that converge faster if features are on similar scales.
# - Prevents Salary (range ~135,000) from dominating Age (range ~42) during weight updates.
# Output: StandardScaler object created.
sc = StandardScaler()

# fit_transform(X_train): 
# 1. fit: Calculates mean and std dev of X_train.
# 2. transform: Applies the scaling formula.
# Why: Compute the scaling parameters on training data and apply them.
# Output: X_train now contains values roughly between -3 and 3 (scaled).
X_train = sc.fit_transform(X_train)

# transform(X_test): 
# Uses the mean/std ALREADY computed from X_train.
# CRITICAL: Do NOT fit on X_test. We must treat test data as if we haven't seen it yet.
# Example: If train mean is 40, we subtract 40 from test data too.
# Why: Scale test set using the SAME parameters as training set to maintain valid comparison.
# Output: X_test now contains values roughly between -3 and 3 (scaled).
X_test = sc.transform(X_test)

# ==========================================
# Step 2: Train a Logistic Regression Model
# ==========================================

# LogisticRegression(random_state):
# - random_state=0: Ensures the solver produces the same result each run (some solvers have random components).
# Other useful args (defaults used here):
# - penalty='l2': Regularization type (Ridge).
# - C=1.0: Inverse of regularization strength. Smaller C = Stronger regularization (prevent overfitting).
# Why: Initialize the Logistic Regression model object.
# Output: LogisticRegression object created (untrained).
classifier = LogisticRegression(random_state=0)

# fit(X, y):
# Trains the model.
# X_train: Training attributes (scaled).
# y_train: Target labels.
# It finds the coefficients (weights) that minimize the error (log-loss).
# Why: Train the model on the training data so it learns the relationship between Age/Salary and Purchase.
# Output: The model 'classifier' is now trained and ready to predict.
classifier.fit(X_train, y_train)

# ==========================================
# Step 3: Plot Sigmoid Curve for one feature (Age)
# ==========================================
# We will model probability of purchase based ONLY on Age for this visualization.

# Why: Initialize a new figure/canvas for plotting. 10x6 inches.
# Output: Empty figure created.
plt.figure(figsize=(10, 6))

# Train a simple 1D Logistic Regression for visualization purposes
# Why: Create a separate simplified model just to visualize 1D probability (Age only).
# Output: New LogisticRegression object created.
clf_age = LogisticRegression()
# reshaping (-1, 1):
# - Sklearn models expect a 2D array for features (rows, attributes).
# - A single pandas Series is 1D.
# - reshape(-1, 1) means "Unknown number of rows (infer it), 1 column".
# Example: [18, 20, 25] becomes [[18], [20], [25]]
# Why: Train the simple 1D model on unscaled Age data.
# Output: clf_age is trained on just Age vs Purchased.
clf_age.fit(df[['Age']], df['Purchased']) 

# Generate a range of age values for plotting the curve
# np.linspace(start, stop, num):
# - start: min age in data
# - stop: max age
# - num=300: Generate 300 evenly spaced points between min and max.
# Why: Create a smooth X-axis to plot the S-curve.
# Output: Array of 300 floats (e.g. [18.0, 18.14, ... 59.0]).
X_age_plot = np.linspace(df['Age'].min(), df['Age'].max(), 300).reshape(-1, 1)

# predict_proba(X):
# Returns array of shape (n_samples, n_classes).
# [[prob_class_0, prob_class_1], ...]
# We select [:, 1] to get the probability of Class 1 (Purchased).
# Why: Get the predicted probability for each of the 300 age points.
# Output: Array of 300 probabilities (0 to 1).
y_age_prob = clf_age.predict_proba(X_age_plot)[:, 1]

# Scatter plot of actual data points
# plt.scatter(x, y, ...):
# - color='red': Sets point color.
# - alpha=0.1: Sets transparency (1=opaque, 0=invisible).
#   Why: Plot the actual observations (0 or 1) to compare with the curve. Low alpha helps visual density.
# Output: Dots (0 and 1) appearing on the plot.
plt.scatter(df['Age'], df['Purchased'], color='red', alpha=0.1, label='Data Points')

# plt.plot(x, y, ...):
# - linewidth=2: Thickness of the line.
# - label: Name for the legend.
# Why: Draw the sigmoid line showing probability increasing with age.
# Output: Blue curve appearing on the plot.
plt.plot(X_age_plot, y_age_prob, color='blue', linewidth=2, label='Logistic Regression Sigmoid')

# Why: Add titles and labels for readability.
plt.title('Probability of Purchase vs Age')
plt.xlabel('Age')
plt.ylabel('Probability of Purchase')
# Why: Add a reference line at 0.5 where the default decision boundary is.
# Output: Dotted gray line at y=0.5.
plt.axhline(0.5, color='gray', linestyle='--')
# Why: Show legend to explain what colors mean.
plt.legend()
# Why: Adjust layout to prevent clipping of labels.
plt.tight_layout()
# Why: Render and display the plot window.
plt.show() 

# ==========================================
# Step 4: Tune Threshold Values and Compare Metrics
# ==========================================

# predict_proba for the test set (using the full model with Age and Salary)
# Why: Get probabilities of "Purchased" for the 100 test set users.
# Output: Array of 100 probabilities.
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Why: Define customized thresholds to test sensitivity/specificity trade-offs.
thresholds = [0.3, 0.5, 0.7]

print("\n--- Tuning Thresholds ---")
for thresh in thresholds:
    # Create predictions based on the custom threshold
    # (y_pred_proba > thresh): returns boolean array, .astype(int) converts True/False to 1/0
    # Why: Convert probabilities to binary class labels (0/1) based on current threshold.
    # Output: Array of 0s and 1s.
    y_pred_custom = (y_pred_proba > thresh).astype(int)
    
    # Calculate metrics
    # precision_score: (True Positives) / (True Positives + False Positives)
    # recall_score: (True Positives) / (True Positives + False Negatives)
    # f1_score: Harmonic mean of Precision and Recall.
    # Why: Compute performance metrics to evaluate the model at this threshold.
    # Output: Float values for p, r, f1.
    p = precision_score(y_test, y_pred_custom)
    r = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    
    # Why: Print the metrics.
    # Output: Formatted string with metrics.
    print(f"Threshold: {thresh} | Precision: {p:.2f} | Recall: {r:.2f} | F1-Score: {f1:.2f}")

# ==========================================
# Step 5: Visualize Decision Boundary
# ==========================================

# Create a meshgrid to plot the decision boundary across the 2D plane (Age vs Salary)
# X_set, y_set: local variables for plot data
# X_set, y_set = sc.inverse_transform(X_test), y_test # Inverse transform to plot in original scale for readability?
# Actually, it's easier to plot on scaled data (as the model is trained on scaled data) 
# OR transform the meshgrid points before predicting.
# Let's stick to scaled data for the boundary generation, but we can label axes with original approximations if needed,
# or just plot in scaled space. Plotting in Scaled Space is standard for showing the decision boundary clearly.
# Why: Set up variables specifically for this plot section, using the scaled test data.
# Output: X_set (100, 2) and y_set (100,) arrays, identical to X_test and y_test.
X_set, y_set = X_test, y_test 

# np.meshgrid(x_range, y_range):
# Creates a rectangular grid out of two 1D arrays.
# Returns two 2D arrays (X1, X2) representing the X and Y coordinates of all grid points.
# We use this to color every pixel in the plot background according to the model's prediction.
# Why: Create a dense grid of points covering the entire plot area (background).
# Output: X1 and X2 (matrices of coordinates).
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)
)

# Predict for every point in the meshgrid
# X1.ravel(): Flattens the 2D grid matrix into a 1D array.
# np.array([x, y]).T: Creates a table with 2 columns (Age, Salary) for all grid points.
# .reshape(X1.shape): Reshapes the predictions back to the grid shape for contour plotting.
# Why: Classify every single background pixel to draw the decision regions.
# Output: 2D array of predictions (0s and 1s) matching the grid shape.
prediction_grid = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

# Why: Initialize a new figure for the decision boundary plot.
# Output: Empty figure.
plt.figure(figsize=(10, 6))

# contourf(X, Y, Z, ...):
# Draws filled contours.
# - X, Y: The grid coordinates.
# - Z: The height/value at those coordinates (here, the class 0 or 1).
# - cmap: Color map ('salmon' for 0, 'dodgerblue' for 1).
from matplotlib.colors import ListedColormap
# Why: Fill the plot background with colors representing the model's decision regions (Red for 0, Blue for 1).
# Output: A plot with two distinct colored regions separated by a straight line.
plt.contourf(X1, X2, prediction_grid, alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))

# Why: Set x and y axis limits to match the grid so there are no white borders.
# Output: Plot axes constrained to the data range.
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Scatter plot for the actual test set points
# Why: Loop through each class (0 and 1) to plot their points separately with different colors.
for i, j in enumerate(np.unique(y_set)):
    # Why: Plot only the points belonging to class 'j' (0 or 1).
    # Output: Red dots for class 0, Blue dots for class 1 superimposed on the regions.
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)

# Why: Label axes and title for clarity.
plt.title('Logistic Regression (Test set) - Scaled Units')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
# Why: Add legend to identify points.
plt.legend()
# Why: Adjust layout to ensure labels fit.
plt.tight_layout()
# Why: Render and display the final plot.
plt.show()

# ==========================================
# Step 6: Key Insights
# ==========================================

# Why: Print section header for insights.
# Output: text "-- Key Insights --"
print("\n--- Key Insights ---")
# Why: Explain the linear nature of the model shown in the plot.
# Output: Text explaining decision boundary.
print("1. Decision Boundary: The model separates purchasers from non-purchasers with a linear line.")
print("   In the 2D plot, this implies a linear combination of Age and Salary determines the outcome.")
print("   Users with higher Age and Salary (Upper Right) are more likely to purchase.")
# Why: Explain the trade-off observed in the threshold tuning step.
# Output: Text explaining Precision/Recall trade-off.
print("2. Threshold Impact: Lowering the threshold (e.g., to 0.3) increases Recall (captures more potential buyers)")
print("   but decreases Precision (more false alarms). This is trade-off based on business cost.")
print("   A high threshold (0.7) ensures that if we predict 'Buy', we are very confident (High Precision).")
# Why: Explain the S-curve visualization.
# Output: Text explaining the sigmoid probability transition.
print("3. Sigmoid Curve: The S-curve shows that probability doesn't increase linearly.")
print("   There is a transition zone where a small increase in Age drastically increases purchase probability.")
