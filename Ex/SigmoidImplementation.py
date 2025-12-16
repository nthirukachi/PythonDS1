"""
Problem Statement:
Implement a function in Python that calculates the sigmoid function for a given input or array of inputs. 
Your function should:
1. Accept both single numbers and NumPy arrays as input.
2. Return the sigmoid value(s).
3. Handle potential numerical overflow issues (e.g., for very large negative values).

Steps to Solve:
1. Import NumPy library.
2. Define a function `stable_sigmoid` that takes `x` as input.
3. Logic:
   - Sigmoid formula is 1 / (1 + exp(-x)).
   - Issue: If x is a large negative number (e.g., -1000), exp(-x) becomes exp(1000) which overflows to Infinity.
   - Solution: Use an alternative formula for negative x: exp(x) / (1 + exp(x)). This keeps exponents negative (small), avoiding overflow.
4. Use `np.where` to apply the correct formula based on whether elements are positive or negative.
5. Create test cases for scalar (-100, 0, 2) and array ([-5, 0, 5]) inputs.

Expected Output:
- Input -100: 0.0 (Approaches 0)
- Input 0: 0.5
- Input 2: ~0.880797
- Input Array [-5, 0, 5]: [0.00669, 0.5, 0.99331]
"""

# Why: NumPy is the standard library for numerical computing in Python. 
# It provides support for arrays, matrices, and high-level mathematical functions.
# Output: Module 'numpy' is loaded and aliased as 'np'.
import numpy as np

def stable_sigmoid(x):
    """
    Calculates the sigmoid function numerically stably.
    
    Sigmoid Formula: S(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    
    Parameters:
    x (int, float, list, np.ndarray): Input value(s) to apply sigmoid to.
    
    Returns:
    np.ndarray or float: Sigmoid output between 0 and 1.
    """
    
    # Why: We need to ensure the input 'x' is in a format NumPy can work with (an array).
    # Even if the user provides a list [1, 2] or a scalar 5, np.array(x) handles it.
    # Output: 'x' is converted to a NumPy array object.
    x = np.array(x)
    
    # Why: Handle Numerical Overflow for very large negative numbers.
    # Standard Formula: 1 / (1 + e^-x)
    # Problem: If x = -1000, then e^-(-1000) = e^1000 -> Overflow to Infinity. 1/(1+inf) = 0.
    # While valid, it triggers RuntimeWarnings.
    # Stable Implementation:
    # For x >= 0: Use 1 / (1 + e^-x)
    # For x < 0: Use e^x / (1 + e^x). (Multiplying numerator and denominator by e^x).
    # This ensures we always compute e^(negative), which approaches 0, never Infinity.
    
    # np.where(condition, x, y):
    # - condition: x >= 0
    # - x (if True): 1 / (1 + np.exp(-x))
    # - y (if False): np.exp(x) / (1 + np.exp(x))
    # Why: Apply one formula to positive elements and the other to negative elements efficiently.
    # Output: An array of sigmoid values with the same shape as input x.
    result = np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )
    
    # Why: If the original input was a scalar (e.g., 5.0), we want to return a scalar (FLOAT), not a 0-d array.
    # result.item(): Converts a size-1 array to a Python standard scalar.
    # Output: Returns float if input was scalar, else returns np.ndarray.
    if result.ndim == 0:
        return result.item()
        
    return result

# ==========================================
# Test Cases
# ==========================================

print("--- Testing Scalar Inputs ---")

# Test Case 1: Large Negative Value
# Why: Test stability. -100 should be extremely close to 0.
# Input: -100
val1 = -100
print(f"Input: {val1}")
out1 = stable_sigmoid(val1)
print(f"Sigmoid: {out1}")
# Expected Output: ~0.0 (actually something like 3.72e-44)

# Test Case 2: Zero
# Why: The inflection point of Sigmoid.
# Input: 0
val2 = 0
print(f"Input: {val2}")
out2 = stable_sigmoid(val2)
print(f"Sigmoid: {out2}")
# Expected Output: 0.5

# Test Case 3: Positive Value
# Why: Standard positive input.
# Input: 2
val3 = 2
print(f"Input: {val3}")
out3 = stable_sigmoid(val3)
print(f"Sigmoid: {out3}")
# Expected Output: ~0.880797

print("\n--- Testing Array Input ---")

# Test Case 4: Array
# Why: Verify the function handles lists/arrays correctly using vectorization.
# Input: [-5, 0, 5]
val_arr = np.array([-5, 0, 5])
print(f"Input Array: {val_arr}")
out_arr = stable_sigmoid(val_arr)
print(f"Sigmoid Array: {out_arr}")
# Expected Output: [0.00669285 0.5        0.99330715]
