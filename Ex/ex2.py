# Import the NumPy library and alias it as 'np'.
# Why: NumPy is the fundamental package for scientific computing in Python. It provides support for arrays (matrices), which are more efficient than Python lists.
import numpy as np

# Import the 'pyplot' module from the Matplotlib library and alias it as 'plt'.
# Why: Matplotlib is a plotting library. 'pyplot' provides a MATLAB-like interface for creating figures and axes.
import matplotlib.pyplot as plt

# Import the 'make_classification' function from the 'sklearn.datasets' module.
# Why: This function is used to generate a random n-class classification dataset. It's great for testing algorithms without needing real data.
from sklearn.datasets import make_classification

# Import the 'LogisticRegression' class from the 'sklearn.linear_model' module.
# Why: Logistic Regression is a linear model for classification (predicting categories, e.g., Yes/No, 0/1).
from sklearn.linear_model import LogisticRegression

# Generate a synthetic 2D dataset using 'make_classification'.
# The function returns two arrays: 
#   - X: The input features (the data points).
#   - y: The target labels (the class each point belongs to, e.g., 0 or 1).
X, y = make_classification(
    n_samples=200,          # The total number of points (rows) to generate. Example: 200 means 200 data points. A larger value like 1000 gives more data.
    n_features=2,           # The total number of features (columns/dimensions) for each point. Example: 2 means each point has an x and y coordinate.
    n_redundant=0,          # The number of redundant features. These are linear combinations of informative features. 0 means no useless repeated info.
    n_clusters_per_class=1, # The number of clusters per class. Example: 1 means each class (0 and 1) is grouped in one main blob.
    random_state=42         # Controls the shuffling applied to the data. Why: Setting a specific number (seed) ensures the result is reproducible (same numbers every time you run it).
)

# Initialize the Logistic Regression model.
# Why: This creates an instance of the model with default parameters. You can tune it (e.g., LogisticRegression(C=0.5)) to change regularization strength.
model = LogisticRegression()

# Train the model using the generated dataset.
# The 'fit' method adjusts the model weights to minimize the error between predictions and actual labels 'y'.
# Arguments:
#   - X: The training data (features), shape (n_samples, n_features).
#   - y: The target values (labels), shape (n_samples,).
model.fit(X, y)

# Create a meshgrid to plot the decision boundary.
# A meshgrid is a rectangular grid of values that covers the range of the data.
# First, determine the minimum and maximum values for the first feature (column 0 of X).
# We subtract/add 1 to add some padding around the edges of the plot.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

# Determine the minimum and maximum values for the second feature (column 1 of X).
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Generate the grid coordinates 'xx' and 'yy'.
# np.linspace(start, stop, num) generates 'num' evenly spaced samples calculated over the interval [start, stop].
# Example: np.linspace(0, 10, 5) -> array([0., 2.5, 5., 7.5, 10.])
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300), # Generate 300 points along the x-axis range.
    np.linspace(y_min, y_max, 300)  # Generate 300 points along the y-axis range.
)

# Predict probabilities for every point on the grid.
# Why: We want to see what the model thinks about every possible point in the background, not just the original data points.
# np.c_[...] stacks 1D arrays as columns into a 2D array.
# xx.ravel() flattens the 300x300 grid into a single long array of 90,000 points.
# model.predict_proba(...) returns probability estimates for all classes. We take [:, 1] to get the probability of Class 1.
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Reshape the result 'Z' back to the original grid shape (300, 300) so it can be plotted as a surface/contour.
Z = Z.reshape(xx.shape)

# Plot the decision regions using filled contours.
# plt.contourf draws filled contours.
# Arguments:
#   - xx, yy: The coordinate positions.
#   - Z >= 0.5: The height values. Here, we create a boolean mask (True if prob >= 0.5, else False) to visually separate the two classes (0 and 1).
#   - alpha=0.3: The transparency level (0 is transparent, 1 is opaque). Why: Makes the background faint so we can see the data points on top.
plt.contourf(xx, yy, Z >= 0.5, alpha=0.3)

# Plot the actual data points.
# plt.scatter draws points (markers).
# X[y == 0] filters the rows of X where the label y is 0.
# [:, 0] selects the first feature (x-coordinate), and [:, 1] selects the second feature (y-coordinate).
# label="Class 0": naming it for the legend.
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0")

# Plot the points for Class 1 similarly.
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1")

# Draw the decision boundary where the probability is exactly 0.5.
# plt.contour draws contour lines (not filled).
# levels=[0.5]: We only want to draw the line where Z is 0.5.
# linewidths=2: Makes the line thicker to be visible.
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2)

# Set the label for the x-axis.
plt.xlabel("Feature 1")

# Set the label for the y-axis.
plt.ylabel("Feature 2")

# Set the title of the plot.
plt.title("Logistic Regression Decision Boundary")

# Show the legend (uses the 'label' arguments from plt.scatter calls).
plt.legend()

# Display the plot to the screen.
# Why: Without this, the plot is created in memory but might not appear.
plt.show()
