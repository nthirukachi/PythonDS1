"""
ONEHOTENCODER WITH SPARSE_OUTPUT EXPLAINED
Complete example with detailed line-by-line comments
"""

# Import pandas library for creating and manipulating DataFrames
import pandas as pd

# Import OneHotEncoder from scikit-learn preprocessing module
# OneHotEncoder converts categorical variables to binary dummy variables
from sklearn.preprocessing import OneHotEncoder

# Import numpy for numerical operations and array manipulation
import numpy as np

# =====================================================================
# EXAMPLE 1: SPARSE_OUTPUT=TRUE (Sparse Matrix - Memory Efficient)
# =====================================================================

print("=" * 80)
print("EXAMPLE 1: SPARSE_OUTPUT=TRUE")
print("=" * 80)

# Create a sample DataFrame with categorical columns
# This represents product data with two categorical features: Color and Size
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],    # Categorical: Colors
    'Size': ['S', 'M', 'L', 'M', 'S']                     # Categorical: Sizes
})

# Display the original DataFrame before encoding
print("\nOriginal Data:")
print(data)
# Shows: 5 rows with Color and Size categories

# Create OneHotEncoder object with sparse_output=True
# sparse_output=True returns a sparse matrix (scipy sparse format)
# This saves memory by only storing non-zero values (the 1s)
# handle_unknown='ignore' means: if new categories appear, ignore them instead of error
encoder_sparse = OneHotEncoder(sparse_output=True, handle_unknown='ignore')

# fit_transform() does two things:
# 1. fit() - learns all unique categories from the data (Red, Blue, Green, S, M, L)
# 2. transform() - converts the categories to binary columns
# Result is a sparse matrix where each column represents a category
encoded_sparse = encoder_sparse.fit_transform(data[['Color', 'Size']])

# Print the data type of encoded_sparse
# With sparse_output=True, it's a scipy sparse matrix (memory-efficient)
print(f"\nType with sparse_output=True: {type(encoded_sparse)}")
# Output: <class 'scipy.sparse._matrix.csr_matrix'>

# Print the shape of the encoded data
# 5 rows (original data rows), 5 columns (unique categories: Red, Blue, Green, S, M)
print(f"Shape: {encoded_sparse.shape}")
# Output: (5, 5)

# Print the sparse matrix representation
# Shows only non-zero values as (row_index, col_index) position with value
print("\nSparse Matrix (only shows non-zero positions):")
print(encoded_sparse)
# Example output shows (row, col) pairs where value is 1

# =====================================================================
# EXAMPLE 2: SPARSE_OUTPUT=FALSE (Dense Array - Easy to Use)
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: SPARSE_OUTPUT=FALSE")
print("=" * 80)

# Create another OneHotEncoder with sparse_output=False
# This returns a regular numpy array (dense format)
# Dense format shows all values (0s and 1s) in a readable grid
encoder_dense = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# fit_transform() learns categories and transforms data into dense array
# Result is a 2D numpy array where:
# - Each row = one original data sample
# - Each column = one category (binary: 0 or 1)
encoded_dense = encoder_dense.fit_transform(data[['Color', 'Size']])

# Print the data type of encoded_dense
# With sparse_output=False, it's a regular numpy array
print(f"\nType with sparse_output=False: {type(encoded_dense)}")
# Output: <class 'numpy.ndarray'>

# Print the shape (same as sparse version)
# 5 rows and 5 columns
print(f"Shape: {encoded_dense.shape}")
# Output: (5, 5)

# Print the actual encoded array
# All values visible: 1 for present category, 0 for absent category
print("\nDense Array (shows all values - zeros and ones):")
print(encoded_dense)
# Output: 5x5 matrix with 0s and 1s

# =====================================================================
# EXAMPLE 3: GET FEATURE NAMES FOR BETTER UNDERSTANDING
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: FEATURE NAMES")
print("=" * 80)

# Get the names of the encoded features from the encoder
# Returns array of feature names like ['Color_Red', 'Color_Blue', etc.]
# These names tell us what each column represents
feature_names = encoder_dense.get_feature_names_out(['Color', 'Size'])

# Print the feature names
# Shows the mapping: which column represents which category
print("\nFeature Names (what each column represents):")
print(feature_names)
# Output: ['Color_Blue' 'Color_Green' 'Color_Red' 'Size_M' 'Size_S']

# =====================================================================
# EXAMPLE 4: CONVERT TO DATAFRAME (Most Useful!)
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: CONVERT TO DATAFRAME")
print("=" * 80)

# Create a pandas DataFrame from the encoded dense array
# This is the most practical format for further analysis
# Parameters:
#   - encoded_dense: the 2D numpy array with encoded values
#   - columns=feature_names: assign meaningful column names
#   - index=data.index: preserve the original row indices for alignment
df_encoded = pd.DataFrame(
    encoded_dense,           # The encoded binary data from OneHotEncoder
    columns=feature_names,   # Column names like 'Color_Red', 'Size_M', etc.
    index=data.index         # Use same indices as original data for alignment
)

# Display the encoded DataFrame
# Much more readable than arrays!
print("\nEncoded as DataFrame (Most Readable!):")
print(df_encoded)
# Output:
#   Color_Blue  Color_Green  Color_Red  Size_M  Size_S
# 0         0.0          0.0        1.0     0.0     1.0
# 1         1.0          0.0        0.0     1.0     0.0
# etc.

# =====================================================================
# EXAMPLE 5: WHY sparse_output=False IS USED IN YOUR CODE
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: WHY USE sparse_output=False")
print("=" * 80)

# Reason 1: Easy conversion to DataFrame
# sparse_output=False gives numpy array that converts directly to DataFrame
print("\nReason 1: Easy DataFrame Conversion")
print("With sparse_output=False:")
df_example1 = pd.DataFrame(encoded_dense, columns=feature_names)
print("✓ Direct conversion works perfectly")

print("\nWith sparse_output=True:")
print("✗ Need to convert first: sparse_matrix.toarray()")
print("✓ Then convert to DataFrame")

# Reason 2: Compatibility with pandas operations
# Dense arrays work seamlessly with pandas methods
print("\n\nReason 2: Pandas Compatibility")
print("Dense arrays support all pandas operations:")
print("- .mean(), .describe(), .fillna()")
print("- .concat(), .merge(), .groupby()")
print("- Indexing: df[['Color_Red']], df.iloc[0]")

# Reason 3: California Housing dataset is small enough
# 20,640 rows × number of categories = manageable memory
print("\n\nReason 3: Dataset Size")
print("California Housing: 20,640 rows")
print("Memory usage: ~165 KB for encoded data (small)")
print("sparse_output=False is perfectly fine!")

# =====================================================================
# EXAMPLE 6: LINE-BY-LINE BREAKDOWN OF YOUR CODE
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: YOUR CALIFORNIA HOUSING CODE EXPLAINED")
print("=" * 80)

# This is exactly how it's used in california_housing_analysis.py:

print("\nStep 1: Create encoder object")
# OneHotEncoder() - creates the encoder
# sparse_output=False - returns numpy array (not sparse matrix)
# handle_unknown='ignore' - if test data has new categories, ignore them gracefully
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
print("✓ Encoder created with sparse_output=False")

print("\nStep 2: Fit and transform categorical features")
# encoder.fit_transform(features[categorical_columns])
# - Learns all unique categories from training data
# - Converts each category to binary columns
# - Returns 2D numpy array of shape (n_samples, n_categories)
encoded_categorical = encoder.fit_transform(data[['Color', 'Size']])
print(f"✓ Data encoded: {encoded_categorical.shape}")

print("\nStep 3: Get feature names for readability")
# encoder.get_feature_names_out(categorical_columns)
# - Returns array of feature names
# - Makes it clear which column represents which category
# - Example: ['Color_Blue', 'Color_Red', 'Size_M', 'Size_S']
categorical_feature_names = encoder.get_feature_names_out(['Color', 'Size'])
print(f"✓ Feature names: {categorical_feature_names}")

print("\nStep 4: Convert to DataFrame for alignment")
# pd.DataFrame(encoded_categorical, columns=..., index=...)
# - Converts numpy array to pandas DataFrame
# - columns: assign meaningful names from encoder
# - index: preserve original indices for proper data alignment
# - This ensures encoded features align with original and other features
df_encoded_categorical = pd.DataFrame(
    encoded_categorical,           # The encoded numpy array
    columns=categorical_feature_names,  # Names from encoder
    index=data.index              # Original indices
)
print(f"✓ DataFrame created: {df_encoded_categorical.shape}")
print(df_encoded_categorical)

# =====================================================================
# EXAMPLE 7: MEMORY AND PERFORMANCE COMPARISON
# =====================================================================

print("\n" + "=" * 80)
print("EXAMPLE 7: SPARSE VS DENSE COMPARISON")
print("=" * 80)

# Calculate memory usage for sparse matrix
sparse_memory = encoded_sparse.data.nbytes  # Only non-zero elements
print(f"\nMemory Usage:")
print(f"Sparse matrix: {sparse_memory} bytes (only stores 1s)")

# Calculate memory usage for dense array
dense_memory = encoded_dense.nbytes  # All elements
print(f"Dense array:   {dense_memory} bytes (stores 0s and 1s)")

# Percentage difference
percentage = (dense_memory - sparse_memory) / sparse_memory * 100
print(f"Dense uses {percentage:.1f}% more memory (for this small example)")

# Practical insight
print(f"\nWhen to use which:")
print(f"- sparse_output=True:  Large datasets with many categories")
print(f"- sparse_output=False: Small-medium datasets, pandas workflows")
print(f"\nYour choice: sparse_output=False ✓ (Correct for California Housing)")

# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = """
ONEHOTENCODER PARAMETERS EXPLAINED:

1. sparse_output=False
   - Returns dense numpy array
   - Shows all 0s and 1s explicitly
   - Easy to view and convert to DataFrame
   - Uses more memory but readable
   - Perfect for your dataset size

2. handle_unknown='ignore'
   - If new category appears in test data, handle gracefully
   - Alternative: 'error' would raise an exception
   - Important for production models

3. get_feature_names_out()
   - Returns meaningful column names
   - Example: 'Color_Red' instead of column 2
   - Critical for interpretable results

WHY EACH STEP IS IMPORTANT:
1. Create encoder - Defines transformation rules
2. fit_transform() - Learn categories and apply them
3. Get feature names - Make output interpretable
4. Convert to DataFrame - Align with other features
5. Set index - Ensure proper data alignment

RESULT:
All categorical data converted to numerical (0s and 1s)
Ready for machine learning models!
"""
print(summary)
