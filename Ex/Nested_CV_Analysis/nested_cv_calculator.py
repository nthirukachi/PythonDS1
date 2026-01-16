"""
Problem Statement:
    Nested CV compute + honest estimation: You have a dataset of 12,000 samples and plan to estimate performance using nested cross-validation.
    The outer loop uses 5 folds. Inside each outer-training split, you run an inner 4-fold CV grid search over 18 hyperparameter combinations.
    You train 1 model per fold per hyperparameter combo. Each training run takes 7 minutes on your GPU.
    Additionally, after selecting the best hyperparameters for each outer fold, you retrain once on the full outer-training split (so 1 extra training run per outer fold) which takes 10 minutes each.

    (a) Compute the total number of training runs.
    (b) Compute total training time in hours.
    (c) If you reduce the grid from 18 to 10 combinations, recompute (a) and (b) and report the % time saved.

Steps to Solve the Problem:
    1.  Define the parameters for the Nested Cross-Validation (Outer folds, Inner folds, Hyperparameter combinations).
    2.  Define the computational costs (Training time per fold, Refit time per outer fold).
    3.  Calculate sub-problems for Scenario A (18 combinations):
        -   Calculate Inner Loop Runs: (Outer Folds) * (Inner Folds) * (Hyperparameter Combinations).
        -   Calculate Refit Runs: (Outer Folds) * 1.
        -   Calculate Total Runs: Inner Loop Runs + Refit Runs.
        -   Calculate Inner Loop Time: Inner Loop Runs * Training Time per Run.
        -   Calculate Refit Time: Refit Runs * Refit Time per Run.
        -   Calculate Total Time: Inner Loop Time + Refit Time.
        -   Convert Total Time to Hours.
    4.  Calculate sub-problems for Scenario B (10 combinations) using the same logic as step 3.
    5.  Compare Scenario A and Scenario B:
        -   Calculate Time Difference.
        -   Calculate Percentage Time Saved.
    6.  Output all results clearly.

Expected Output of the Problem:
    -   Total runs for Scenario A (18 combos).
    -   Total time in hours for Scenario A.
    -   Total runs for Scenario B (10 combos).
    -   Total time in hours for Scenario B.
    -   Percentage of time saved by reducing combinations.

Explanation:
    This script automates the arithmetic required to estimate the computational budget for a Nested Cross-Validation experiment.
    It breaks down the total effort into 'Inner Loop' (Grid Search) and 'Outer Loop Refit' components.
    By parameterizing the number of combinations, we can easily perform "what-if" analysis like comparing 18 vs 10 combinations.
"""

def calculate_nested_cv_cost(outer_folds, inner_folds, param_combos, train_time_min, refit_time_min):
    """
    Detailed explanation of the arguments of the method:

    1. outer_folds:
        3.1: what the argument does: Represents the number of splits in the outer cross-validation loop.
        3.2: why it is used: To define how many times the entire estimation process is repeated to get a robust error estimate.
        3.3: when to used: When setting up the Nested CV structure (k-fold).
        3.4: where to use: In the calculation of total inner runs and total refit runs.
        3.5: How to use: Pass an integer, e.g., 5.

    2. inner_folds:
        3.1: what the argument does: Represents the number of splits in the inner cross-validation loop used for hyperparameter tuning.
        3.2: why it is used: To validate hyperparameter performance within each outer training set.
        3.3: when to used: When defining the grid search strategy inside the nested loop.
        3.4: where to use: In the calculation of total inner runs.
        3.5: How to use: Pass an integer, e.g., 4.

    3. param_combos:
        3.1: what the argument does: The number of candidate hyperparameter sets to evaluate.
        3.2: why it is used: Determines the size of the search space for the best model configuration.
        3.3: when to used: When defining the grid size (e.g., 18 vs 10).
        3.4: where to use: Multiplier for the inner loop runs.
        3.5: How to use: Pass an integer, e.g., 18.

    4. train_time_min:
        3.1: what the argument does: The time taken to train a single model on one fold.
        3.2: why it is used: To estimate the temporal cost of the grid search.
        3.3: when to used: When hardware benchmarks are known (e.g., 7 mins per run).
        3.4: where to use: Multiplier for total inner loop time.
        3.5: How to use: Pass a float or int in minutes, e.g., 7.

    5. refit_time_min:
        3.1: what the argument does: The time taken to retrain the best model on the full outer training set.
        3.2: why it is used: To account for the final model training cost in each outer fold.
        3.3: when to used: Typically longer than fold training as the data size is larger.
        3.4: where to use: Multiplier for total refit time.
        3.5: How to use: Pass a float or int in minutes, e.g., 10.

    Sample Example:
        calculate_nested_cv_cost(5, 4, 18, 7, 10)
    """

    # 2.1: what the line of code does: Calculates total inner runs by multiplying outer folds, inner folds, and parameter combos.
    # 2.2: why it is used: To find the total number of models trained during the grid search phase across all outer folds.
    # 2.3: when to used: First step of cost estimation.
    # 2.4: where to use: Inside the calculation function.
    # 2.5: How to use: Simple multiplication of integer variables.
    # 2.6: How it works: 5 outer * 4 inner * 18 params = 360 runs.
    # 2.7: Output: 360 (for the 18-combo case).
    total_inner_runs = outer_folds * inner_folds * param_combos
    # Expected Output: 360

    # 2.1: what the line of code does: Calculates the number of refit runs, which equals the number of outer folds.
    # 2.2: why it is used: Because after each inner grid search, we retrain the best model once on that outer fold's training data.
    # 2.3: when to used: After determining grid search volume.
    # 2.4: where to use: In total run calculation.
    # 2.5: How to use: Assign outer_folds value to a new descriptive variable.
    # 2.6: How it works: 5 outer folds implies 5 winning configurations to retrain.
    # 2.7: Output: 5.
    total_refit_runs = outer_folds
    # Expected Output: 5

    # 2.1: what the line of code does: Sums inner runs and refit runs to get the grand total of training processes.
    # 2.2: why it is used: To answer part (a) of the requirement: "Compute the total number of training runs".
    # 2.3: when to used: Final step for run counting.
    # 2.4: where to use: Return value or intermediate calculation.
    # 2.5: How to use: Addition operator.
    # 2.6: How it works: 360 + 5 = 365.
    # 2.7: Output: 365.
    total_runs = total_inner_runs + total_refit_runs
    # Expected Output: 365

    # 2.1: what the line of code does: Calculates time spent in inner loops by multiplying runs by time-per-run.
    # 2.2: why it is used: To quantify the GPU time consumed by grid search.
    # 2.3: when to used: For time estimation.
    # 2.4: where to use: In total time calculation.
    # 2.5: How to use: Multiplication.
    # 2.6: How it works: 360 runs * 7 mins = 2520 mins.
    # 2.7: Output: 2520.
    inner_time_total = total_inner_runs * train_time_min
    # Expected Output: 2520

    # 2.1: what the line of code does: Calculates time spent refitting models.
    # 2.2: why it is used: To quantify the GPU time consumed by final model retraining.
    # 2.3: when to used: For time estimation.
    # 2.4: where to use: In total time calculation.
    # 2.5: How to use: Multiplication.
    # 2.6: How it works: 5 runs * 10 mins = 50 mins.
    # 2.7: Output: 50.
    refit_time_total = total_refit_runs * refit_time_min
    # Expected Output: 50

    # 2.1: what the line of code does: Sums inner time and refit time for total minutes.
    # 2.2: why it is used: Total duration in minutes before conversion.
    # 2.3: when to used: Before converting to hours.
    # 2.4: where to use: Intermediate time calculation.
    # 2.5: How to use: Addition.
    # 2.6: How it works: 2520 + 50 = 2570.
    # 2.7: Output: 2570.
    total_time_min = inner_time_total + refit_time_total
    # Expected Output: 2570

    # 2.1: what the line of code does: Converts total minutes to hours.
    # 2.2: why it is used: To answer part (b) of the requirement "in hours".
    # 2.3: when to used: Final result formatting.
    # 2.4: where to use: Return value.
    # 2.5: How to use: Division by 60.
    # 2.6: How it works: 2570 / 60 = 42.8333...
    # 2.7: Output: 42.833333333333336.
    total_time_hours = total_time_min / 60
    # Expected Output: ~42.83

    # 2.1: what the line of code does: Returns a dictionary containing all computed metrics.
    # 2.2: why it is used: To pack multiple results into a single object for easy access.
    # 2.3: when to used: At the end of the function.
    # 2.4: where to use: Function exit.
    # 2.5: How to use: Dictionary literal construction.
    # 2.6: How it works: Maps string keys to calculated values.
    # 2.7: Output: {'runs': 365, 'hours': 42.83, 'minutes': 2570}.
    return {
        "runs": total_runs,
        "hours": total_time_hours,
        "minutes": total_time_min
    }
    # Expected Output: Dict object

# 2.1: what the line of code does: Sets the variable for number of outer folds to 5.
# 2.2: why it is used: Requirement from problem statement "outer loop uses 5 folds".
# 2.3: when to used: Initialization of constants.
# 2.4: where to use: Global scope or main execution block.
# 2.5: How to use: Assignment.
# 2.6: How it works: Stores integer 5 in memory.
# 2.7: Output: None (Assignment).
OUTER_FOLDS = 5
# Expected Output: 5

# 2.1: what the line of code does: Sets the variable for number of inner folds to 4.
# 2.2: why it is used: Requirement from problem statement "inner 4-fold CV".
# 2.3: when to used: Initialization of constants.
# 2.4: where to use: Global scope or main execution block.
# 2.5: How to use: Assignment.
# 2.6: How it works: Stores integer 4.
# 2.7: Output: None.
INNER_FOLDS = 4
# Expected Output: 4

# 2.1: what the line of code does: Sets training time per fold in minutes.
# 2.2: why it is used: Requirement "Each training run takes 7 minutes".
# 2.3: when to used: Initialization.
# 2.4: where to use: Global scope.
# 2.5: How to use: Assignment.
# 2.6: How it works: Stores integer 7.
# 2.7: Output: None.
TRAIN_TIME_MIN = 7
# Expected Output: 7

# 2.1: what the line of code does: Sets refit time per outer fold in minutes.
# 2.2: why it is used: Requirement "retrain... takes 10 minutes".
# 2.3: when to used: Initialization.
# 2.4: where to use: Global scope.
# 2.5: How to use: Assignment.
# 2.6: How it works: Stores integer 10.
# 2.7: Output: None.
REFIT_TIME_MIN = 10
# Expected Output: 10

# --- Scenario A: 18 Combinations ---

# 2.1: what the line of code does: Calls the calculation function for the 18-parameter case.
# 2.2: why it is used: To compute values for part (a) and (b).
# 2.3: when to used: Analysis phase.
# 2.4: where to use: Main script body.
# 2.5: How to use: Call with 18 as the param_combos argument.
# 2.6: How it works: Executes logical steps defined in function.
# 2.7: Output: {'runs': 365, 'hours': 42.83, ...}.
result_a = calculate_nested_cv_cost(OUTER_FOLDS, INNER_FOLDS, 18, TRAIN_TIME_MIN, REFIT_TIME_MIN)
# Expected Output: Result dict

# 2.1: what the line of code does: Prints the header for Scenario A results.
# 2.2: why it is used: For clear output formatting.
# 2.3: when to used: Before displaying numbers.
# 2.4: where to use: Standard output.
# 2.5: How to use: print() function.
# 2.6: How it works: Writes string to console.
# 2.7: Output: "--- Scenario A (18 Combinations) ---".
print("--- Scenario A (18 Combinations) ---")
# Expected Output: Print Header

# 2.1: what the line of code does: Prints total runs for Scenario A.
# 2.2: why it is used: Reporting answer (a).
# 2.3: when to used: After calculation.
# 2.4: where to use: Standard output.
# 2.5: How to use: F-string formatting.
# 2.6: How it works: Accesses 'runs' key from dictionary.
# 2.7: Output: "Total Training Runs: 365".
print(f"Total Training Runs: {result_a['runs']}")
# Expected Output: "Total Training Runs: 365"

# 2.1: what the line of code does: Prints total hours for Scenario A.
# 2.2: why it is used: Reporting answer (b).
# 2.3: when to used: After calculation.
# 2.4: where to use: Standard output.
# 2.5: How to use: F-string formatting with rounding.
# 2.6: How it works: Accesses 'hours' key and rounds to 2 decimals.
# 2.7: Output: "Total Training Time: 42.83 hours".
print(f"Total Training Time: {result_a['hours']:.2f} hours")
# Expected Output: "Total Training Time: 42.83 hours"

# --- Scenario B: 10 Combinations ---

# 2.1: what the line of code does: Calls the calculation function for the 10-parameter case.
# 2.2: why it is used: To compute values for part (c).
# 2.3: when to used: Analysis phase for reduced grid.
# 2.4: where to use: Main script body.
# 2.5: How to use: Call with 10 as the param_combos argument.
# 2.6: How it works: Executes logical steps defined in function.
# 2.7: Output: {'runs': 205, 'hours': 24.17, ...}.
result_b = calculate_nested_cv_cost(OUTER_FOLDS, INNER_FOLDS, 10, TRAIN_TIME_MIN, REFIT_TIME_MIN)
# Expected Output: Result dict

# 2.1: what the line of code does: Prints the header for Scenario B results.
# 2.2: why it is used: For clear output formatting.
# 2.3: when to used: Before displaying numbers.
# 2.4: where to use: Standard output.
# 2.5: How to use: print() function.
# 2.6: How it works: Writes string to console.
# 2.7: Output: "\n--- Scenario B (10 Combinations) ---".
print("\n--- Scenario B (10 Combinations) ---")
# Expected Output: Print Header

# 2.1: what the line of code does: Prints total runs for Scenario B.
# 2.2: why it is used: Reporting answer (c) - part (a) recomputed.
# 2.3: when to used: After calculation.
# 2.4: where to use: Standard output.
# 2.5: How to use: F-string formatting.
# 2.6: How it works: Accesses 'runs' key.
# 2.7: Output: "Total Training Runs: 205".
print(f"Total Training Runs: {result_b['runs']}")
# Expected Output: "Total Training Runs: 205"

# 2.1: what the line of code does: Prints total hours for Scenario B.
# 2.2: why it is used: Reporting answer (c) - part (b) recomputed.
# 2.3: when to used: After calculation.
# 2.4: where to use: Standard output.
# 2.5: How to use: F-string formatting.
# 2.6: How it works: Accesses 'hours' key.
# 2.7: Output: "Total Training Time: 24.17 hours".
print(f"Total Training Time: {result_b['hours']:.2f} hours")
# Expected Output: "Total Training Time: 24.17 hours"

# --- Time Saved ---

# 2.1: what the line of code does: Calculates the percentage of time saved.
# 2.2: why it is used: To answer the final part of (c) "report the % time saved".
# 2.3: when to used: After both scenarios are computed.
# 2.4: where to use: Main script calculation.
# 2.5: How to use: Formula (Old - New) / Old * 100.
# 2.6: How it works: (42.83 - 24.17) / 42.83 * 100 = ~43.58.
# 2.7: Output: 43.5797...
time_saved_pct = ((result_a['minutes'] - result_b['minutes']) / result_a['minutes']) * 100
# Expected Output: ~43.58

# 2.1: what the line of code does: Prints the percentage of time saved.
# 2.2: why it is used: Reporting the final metric.
# 2.3: when to used: Final output.
# 2.4: where to use: Standard output.
# 2.5: How to use: F-string with rounding.
# 2.6: How it works: Prints the float value formatted to 2 decimals.
# 2.7: Output: "Percentage Time Saved: 43.58%".
print(f"Percentage Time Saved: {time_saved_pct:.2f}%")
# Expected Output: "Percentage Time Saved: 43.58%"
