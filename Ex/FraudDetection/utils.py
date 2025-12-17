"""
utils.py
-------------------------------------------------------------------------------
PROBLEM STATEMENT:
The Fraud Detection System requires a centralized way to load the dataset and 
visualize results consistently across multiple scripts.

STEPS TO SOLVE:
1. Define the file path for the dataset.
2. Implement a function `load_data()` that safe-guards against missing files.
3. Implement a function `plot_confusion_matrix()` for standard plotting.

SUB-PROBLEM:
- Handling missing files without crashing the entire program.

EXPECTED OUTPUT:
- `load_data()` returns a Pandas DataFrame containing 284,807 transaction rows.
- `plot_confusion_matrix()` displays a matplotlib figure window.
-------------------------------------------------------------------------------
"""

# ----------------- IMPORTS -----------------

# WHAT: Importing pandas library.
# WHY: Pandas is the standard tool for manipulating tabular data (rows/cols).
# WHEN: Always used for Data Processing tasks.
import pandas as pd

# WHAT: Importing numpy library.
# WHY: Needed for numerical operations (arrays) that scikit-learn uses internally.
import numpy as np

# WHAT: Importing matplotlib.pyplot.
# WHY: The core plotting library in Python. Used to create figures/charts.
import matplotlib.pyplot as plt

# WHAT: Importing seaborn.
# WHY: A wrapper around matplotlib that makes plots prettier (heatmaps, histograms) by default.
import seaborn as sns

# WHAT: Importing os module.
# WHY: Provides functions to interact with the Operating System (checking file paths).
import os

# ----------------- CONFIGURATION -----------------

# WHAT: Defining a constant string variable for the dataset location.
# WHY: Hardcoding the path in one place allows us to change it easily later if the file moves.
# WHEN: At the start of the project setup.
DATA_PATH = r"C:\nagpython\demouv\Ex\creditcard.csv"

# ----------------- FUNCTIONS -----------------

def load_data():
    """
    Loads the Credit Card Fraud dataset into a Pandas DataFrame.
    
    WHAT IS THIS METHOD:
    It reads a CSV file from disk and returns it as a Python object.
    
    WHY IS IT USED:
    Machine Learning algorithms cannot read files directly; they need data in memory (RAM).
    
    WHEN IS IT USED:
    Called at the very beginning of every script (EDA, Training, Tuning).
    
    Returns:
        pd.DataFrame or None: The loaded data if successful, else None.
    """
    print(f"Loading data from {DATA_PATH}...")
    
    # WHAT: Check if the file exists at the specified path.
    # ARGUMENTS: `DATA_PATH` (string) - the absolute path to check.
    # WHY: If we try to read a missing file, Python raises a FileNotFoundError and crashes.
    #      We want to handle this gracefully.
    # EXPECTED OUTPUT: Returns True if file exists, False otherwise.
    if not os.path.exists(DATA_PATH):
        # WHAT: Print error message.
        # WHY: To inform the user why the program is stopping.
        print(f"Error: File not found at {DATA_PATH}")
        print("Please ensure the dataset is downloaded and placed correctly.")
        return None
        
    # WHAT: Read the CSV file.
    # METHOD: pd.read_csv()
    # ARGUMENTS:
    #   - filepath_or_buffer: `DATA_PATH` (string).
    # WHY: Parses the comma-separated values into a structured table.
    # EXPECTED OUTPUT: A DataFrame object with ~284k rows and 31 columns.
    df = pd.read_csv(DATA_PATH)
    
    # WHAT: Print the shape of the data.
    # WHY: Confirmation to the user that load was successful and complete.
    # EXPECTED OUTPUT: "Data Loaded Successfully: 284807 rows, 31 columns"
    print(f"Data Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Helper function to visualize the confusion matrix.
    
    WHAT IS THIS METHOD:
    Takes a numerical confusion matrix and draws it as a colored heatmap.
    
    ARGUMENTS:
    1. cm (numpy.ndarray): A 2x2 matrix [[TN, FP], [FN, TP]].
       - WHY: These 4 numbers define the performance of a classifier.
    2. title (str): The heading of the plot.
       - WHY: To distinguish between different models (e.g., 'KNN' vs 'SVM').
    
    EXPECTED OUTPUT:
    A matplotlib window showing a blue heatmap grid.
    """
    # WHAT: Create a new figure with specific size.
    # ARGUMENTS: figsize=(6, 4) -> Width 6 inches, Height 4 inches.
    plt.figure(figsize=(6, 4))
    
    # WHAT: Draw the heatmap.
    # ARGUMENTS:
    #   - data: `cm` (the matrix).
    #   - annot=True: Writes the actual count number inside each box.
    #   - fmt='d': Formats the number as an Integer 'd' (decimal) instead of scientific notation '1e4'.
    #   - cmap='Blues': Uses a blue color scale (White=Low, Dark Blue=High).
    # WHY: Visual representation makes it easier to spot high False Negatives.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # WHAT: Set axis labels and title.
    # WHY: Good plotting practice for readability.
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # WHAT: Render the plot to the screen.
    # WHEN: At the end of the plotting commands.
    plt.show()

if __name__ == "__main__":
    # WHAT: Entry point check.
    # WHY: Allows us to test `load_data()` by running `python utils.py` directly,
    #      without running the whole logic.
    load_data()
