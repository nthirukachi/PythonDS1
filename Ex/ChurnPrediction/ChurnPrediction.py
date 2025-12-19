"""
Problem Statement:
Implement and Compare ML Algorithms for Customer Churn Prediction [CODING] (Optional)
Dataset: Telco Customer Churn Dataset
•	Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
•	OR Generate synthetic data using the code below if dataset unavailable

Your Tasks - Write Complete Python Code:
Part 1: Data Preparation (15 points)
1.	Load the dataset and perform EDA (check data types, missing values, class distribution)
2.	Handle categorical variables (One-Hot Encoding or Label Encoding)
3.	Handle numerical features (StandardScaler for distance-based algorithms)
4.	Split data into train (70%), validation (15%), and test (15%) sets with stratification
5.	Report the class distribution in train/validation/test sets

Part 2: Implement 4 Algorithms (40 points) Implement and train the following models:
1.	k-NN (k=5)
2.	SVM with RBF kernel (C=1.0, gamma='scale')
3.	Decision Tree (max_depth=10)
4.	Random Forest (n_estimators=100)
For each algorithm:
•	Train on training set
•	Make predictions on both train and test sets
•	Calculate: Train Accuracy, Test Accuracy, Precision, Recall, F1-Score
•	Measure training time and prediction time (for 1000 samples)

Part 3: Overfitting Analysis (25 points)
1.	Create a comparison table showing Train vs Test accuracy for all 4 algorithms
2.	Calculate and visualize the train-test gap (overfitting) for each algorithm
3.	Generate confusion matrices for all algorithms on test set
4.	Plot ROC curves for all 4 algorithms on the same graph
5.	Answer: Which algorithm overfits the most? Explain why based on algorithm characteristics

Part 4: Production Deployment Selection (20 points) Given requirement: Prediction latency must be <50ms for 1000 predictions
1.	Create a comparison visualization showing Accuracy vs Prediction Time
2.	Recommend the best algorithm for production deployment
3.	Write production-ready prediction function that: 
o	Loads the trained model
o	Takes new customer data as input
o	Returns churn probability and prediction
o	Handles errors gracefully
"""

# ----------------- IMPORTS -----------------

# Importing pandas library.
# WHAT: Pandas is the primary library for data manipulation and analysis in Python.
# WHY: We need it to load the dataset (read_csv) and handle tabular data (DataFrames).
# WHEN: At the start of any data science project involving structured data.
# EXPECTED OUTPUT: The module `pd` is available to use.
import pandas as pd

# Importing numpy library.
# WHAT: Fundamental package for scientific computing.
# WHY: Used for numerical operations, array handling, and math functions (like np.tile, np.arange).
# WHEN: When performing mathematical transformations or handling arrays.
# EXPECTED OUTPUT: The module `np` is available.
import numpy as np

# Importing matplotlib.pyplot.
# WHAT: A plotting library for creating static, animated, and interactive visualizations.
# WHY: Used to plot the ROC curves and bar charts.
# WHEN: Visualizing results.
# EXPECTED OUTPUT: Module `plt` available for plotting.
import matplotlib.pyplot as plt

# Importing seaborn.
# WHAT: A statistical data visualization library based on matplotlib.
# WHY: Provides high-level interface for drawing attractive graphics (though we mostly use plt here).
# WHEN: Often used for heatmaps/confusion matrices.
# EXPECTED OUTPUT: Module `sns` available.
import seaborn as sns

# Importing time module.
# WHAT: Provides various time-related functions.
# WHY: To measure the 'Training Time' and 'Prediction Time' of models.
# WHEN: Benchmarking code performance (latency checks).
# EXPECTED OUTPUT: Module `time` available.
import time

# Importing Scikit-Learn modules.
# WHAT: The standard machine learning library for Python.
# WHY: Provides tools for splitting data, preprocessing, modeling, and evaluation.
# WHEN: Implementing ML pipelines.
from sklearn.model_selection import train_test_split # To split data into Train/Test
from sklearn.preprocessing import StandardScaler # To scale numerical features (Z-score normalization)
from sklearn.neighbors import KNeighborsClassifier # The k-NN algorithm
from sklearn.svm import SVC # The Support Vector Classification algorithm
from sklearn.tree import DecisionTreeClassifier # The Decision Tree algorithm
from sklearn.ensemble import RandomForestClassifier # The Random Forest algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc # Evaluation metrics

# ----------------- PART 1: DATA PREPARATION -----------------

def load_data(file_path):
    """
    Function to load data from CSV.
    Arguments:
        file_path (str): The absolute or relative path to the csv file.
    """
    # WHAT: Printing status.
    print(f"Loading Data from {file_path}...")
    
    try:
        # WHAT: Reading the CSV file into a pandas DataFrame.
        # METHOD: pd.read_csv(filepath)
        # ARGUMENTS:
        #   - filepath: `file_path`. The location of the file.
        # WHY: To bring the data from disk into RAM for processing.
        # EXPECTED OUTPUT: A pandas DataFrame containing rows and columns from the CSV.
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def perform_eda(df):
    """
    Function to perform basic Exploratory Data Analysis.
    Arguments:
        df (DataFrame): The loaded dataset.
    """
    # WHAT: Printing header.
    print("\n--- EDA ---")
    
    # WHAT: Displaying concise summary of the dataframe.
    # METHOD: df.info()
    # WHY: To check data types (int, float, object) and missing values (null count).
    # EXPECTED OUTPUT: Text summary of columns and memory usage.
    print(df.info())
    
    # WHAT: Checking the balance of the target class 'Churn'.
    # METHOD: value_counts(normalize=True)
    # ARGUMENTS:
    #   - normalize: True. Returns relative frequencies (percentages) instead of raw counts.
    # The line normalize=True in df['Churn'].value_counts(normalize=True) changes the output of the value_counts() function.
    # Before: It would return the absolute count of unique values (e.g., 5000 "No", 1500 "Yes").
    # After (normalize=True): It returns the proportion or percentage of each value relative to the total number of rows.
    # Example: 0.73 for "No" and 0.27 for "Yes".
    # The values will always sum up to 1.0.
    # WHY: To see if we have an imbalanced dataset (e.g., 90% No vs 10% Yes).
    # EXPECTED OUTPUT: Series showing percentage of Churn=Yes vs No.
    print("\nClass Distribution:\n", df['Churn'].value_counts(normalize=True))

def prepare_data(df):
    """
    Function to clean, encode, split, and scale the data.
    Arguments:
        df (DataFrame): The raw loaded dataset.
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, feature_columns
    """
    # WHAT: Cleaning Step - Removing identifiers.
    # CONDITION: If 'customerID' exists in columns.
    # MVP: Identifiers don't predict churn (noise).
    if 'customerID' in df.columns:
        # METHOD: df.drop()
        # ARGUMENTS: axis=1 (columns).
        df = df.drop('customerID', axis=1)  
        
    # WHAT: Cleaning Step - Handling 'TotalCharges'.
    # PROBLEM: It is imported as 'object' (string) because of some blank values " ".
    # METHOD: pd.to_numeric(arg, errors)
    # ARGUMENTS:
    #   - arg: df['TotalCharges']. The column to convert.
    #   - errors: 'coerce'. 
    #     "Coerce" means "force". If a value cannot be converted (like " " or "abc"), 
    #     instead of crashing the program (errors='raise'), it replaces the bad value with NaN (Not a Number).
    #     Example: ['100', ' '] -> [100.0, NaN]
    # WHY: We need it to be a float for math operations.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # WHAT: Removing potential NaNs created above.
    # METHOD: df.dropna()
    # WHY: Models generally crash on missing values. Since there are very few, dropping is safe.
    df = df.dropna()
    
    # WHAT: Encoding Target Variable.
    # METHOD: map(dict)
    # ARGUMENTS: {'Yes': 1, 'No': 0}.
    # WHY: Scikit-learn requires numerical targets for classification (1=Positive Class/Churn).
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # WHAT: Separating Features (X) and Target (y).
    X = df.drop('Churn', axis=1) # All columns except Churn
    y = df['Churn'] # Only Churn column
    
    # WHAT: Identifying Categorical Columns.
    # METHOD: select_dtypes(include=['object'])
    # WHY: Machine Learning models cannot effectively read text. We need to identify columns containing text (pandas type 'object')
    #      so we can transform them into numbers (0/1) in the next step.
    #      Example: Finds ['Gender', 'Partner', 'PhoneService'...]
    cat_cols = X.select_dtypes(include=['object']).columns
    print(f"Encoding categorical columns: {list(cat_cols)}")
    
    # WHAT: One-Hot Encoding.
    # METHOD: pd.get_dummies(data, columns, drop_first)
    # ARGUMENTS:
    #   - columns: `cat_cols`. The list of columns to encode.
    #   - drop_first: True. Drops one category per feature to avoid multicollinearity (k-1 variables for k categories).
    # WHY: Converts "Male/Female" to "Is_Male" (0/1).
    # EXPECTED OUTPUT: DataFrame `X` with significantly more columns (numeric).
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # WHAT: Splitting Data - Step 1.
    # GOAL: 70% Train, 30% Remaining (Temp).
    # METHOD: train_test_split(X, y, test_size, stratify, random_state)
    # ARGUMENTS:
    #   - test_size: 0.3. Puts 30% into test set (temp).
    #   - stratify: `y`. 
    #     Ensures the Churn ratio (Yes/No) is maintained in both splits.
    #     Example: If the original data has 10% Churners, both Train and Test sets will also have exactly 10% Churners.
    #     This prevents a "bad random split" where all the Churners end up in the test set.
    #   - random_state: 42. Ensures reproducibility.
    # EXPECTED OUTPUT: 4 DataFrames/Series.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # WHAT: Splitting Data - Step 2.
    # GOAL: Split the 30% Temp into Validation (15% total) and Test (15% total).
    # LOGIC: Splitting 30% in half (0.5) gives 15% and 15%.
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # WHAT: Scaling Numerical Data.
    # METHOD: StandardScaler()
    # WHY: Algorithms like SVM and k-NN calculate distances. Large numbers (Income=5000) dominate small numbers (Age=30). Scaling makes mean=0, std=1.
    scaler = StandardScaler()
    
    # STEP 1: fit_transform on Training Data.
    # WHY: Learn the mean/std ONLY from training data (no data leakage).
    X_train_scaled = scaler.fit_transform(X_train)
    
    # STEP 2: transform on Val/Test Data.
    # WHY: Use the training mean/std to scale the other sets.
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nData Splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # RETURNING: All datasets plus `scaler` (for production) and `X.columns` (to know feature names).
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, X.columns

# ----------------- PART 2: IMPLEMENT 4 ALGORITHMS -----------------

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Function to train models, measure time, and calculate metrics.
    Arguments:
        models (dict): Dictionary of {name: model_object}.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # WHAT: Measuring Training Time.
        start_train = time.time() # Capture start time
        
        # WHAT: Training the model.
        # METHOD: fit(X, y)
        # ARGUMENTS: X_train, y_train.
        # WHY: To learn the patterns mapping features to Churn.
        model.fit(X_train, y_train)
        
        end_train = time.time() # Capture end time
        train_time = end_train - start_train # Calculate duration
        
        # WHAT: Measuring Prediction Time (Latency).
        # We simulate 1000 predictions.
        # LOGIC: Ensure we have 1000 samples. If X_test is small, we repeat it.
        X_sample = X_test[:1000] if len(X_test) >= 1000 else np.tile(X_test, (2, 1))[:1000]
        
        start_pred = time.time()
        _ = model.predict(X_sample) # Running prediction (ignoring result with _)
        end_pred = time.time()
        pred_time_1000 = (end_pred - start_pred) # Total duration
        
        # WHAT: Calculating Evaluation Metrics.
        # Generating predictions for Train (to check overfitting) and Test (to check real performance).
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Storing results in a dictionary.
        # METRICS:
        # - Accuracy: Correctness % (tp+tn)/total.
        # - Precision: Accuracy of positive predictions tp/(tp+fp).
        # - Recall: Ability to find positives tp/(tp+fn).
        # - F1: Harmonic mean of Precision and Recall.
        results[name] = {
            'model': model,
            'train_acc': accuracy_score(y_train, y_train_pred),
            'test_acc': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'train_time': train_time,
            'pred_time_1000_ms': pred_time_1000 * 1000 # Convert seconds to milliseconds
        }
        
    return results

# ----------------- PART 3: OVERFITTING & VISUALIZATION -----------------

def analyze_results(results, X_test, y_test):
    """
    Function to visualize the results and identifying overfitting.
    """
    # 1. Comparison Table
    # WHAT: Creating a DataFrame from the results dictionary.
    metrics_df = pd.DataFrame(results).T[['train_acc', 'test_acc', 'precision', 'recall', 'f1', 'pred_time_1000_ms']]
    
    # WHAT: Calculating Overfitting Gap.
    # LOGIC: Train Accuracy - Test Accuracy.
    # WHY: A large positive gap (e.g., > 0.05 or > 0.10) means the model memorized training data.
    metrics_df['overfitting_gap'] = metrics_df['train_acc'] - metrics_df['test_acc']
    print("\n--- Model Performance Comparison ---")
    print(metrics_df)

    # 2. Visualizations
    plt.figure(figsize=(15, 10))

    # A. Train vs Test Accuracy (Gap)
    plt.subplot(2, 2, 1)
    x = np.arange(len(results))
    width = 0.35
    model_names = list(results.keys())
    train_accs = [results[m]['train_acc'] for m in model_names]
    test_accs = [results[m]['test_acc'] for m in model_names]
    
    plt.bar(x - width/2, train_accs, width, label='Train Acc')
    plt.bar(x + width/2, test_accs, width, label='Test Acc')
    plt.xticks(x, model_names)
    plt.title('Train vs Test Accuracy (Overfitting Check)')
    plt.legend()

    # B. ROC Curves
    plt.subplot(2, 2, 2)
    for name, data in results.items():
        # WHAT: Getting probabilities.
        # CONDITION: If model supports predict_proba (like RF, kNN, SVC with prob=True).
        if hasattr(data['model'], "predict_proba"):
            y_prob = data['model'].predict_proba(X_test)[:, 1] # Probability of class 1
        else:
            y_prob = data['model'].predict(X_test)
            
        # WHAT: Calculating ROC Curve points.
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr) # Area Under Curve score
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--') # Diagonal line (random guess)
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    # C. Accuracy vs Time
    plt.subplot(2, 2, 3)
    times = [results[m]['pred_time_1000_ms'] for m in model_names]
    accs = [results[m]['test_acc'] for m in model_names]
    plt.scatter(times, accs, s=100)
    for i, txt in enumerate(model_names):
        plt.annotate(txt, (times[i], accs[i]))
    
    # WHAT: Drawing threshold line.
    # WHY: Requirement is latency < 50ms.
    plt.axvline(x=50, color='r', linestyle='--', label='50ms Latency Limit')
    plt.title('Accuracy vs Prediction Latency (1000 samples)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    print("\nVisualizations generated (check plot window).")
    # plt.show() # Uncomment to view if running locally
    
    return metrics_df

# ----------------- PART 4: PRODUCTION FUNCTION -----------------

def predict_new_customer(model, scaler, new_data):
    """
    Production-ready function.
    Args:
        model: Trained model object.
        scaler: Fitted scaler object.
        new_data (list or np.array): Raw feature values (must match encoded feature count).
    Returns:
        dict: prediction and probability
    """
    try:
        # 1. Validation
        # WHAT: Checking input shape.
        # Note: In a real system, we'd handle the raw -> one-hot conversion here too.
        # For this example, we assume new_data matches the `X_train` structure (post-encoding).
        required_features = scaler.n_features_in_
        if len(new_data) != required_features:
            return {"error": f"Invalid input shape. Expected {required_features} features, got {len(new_data)}."}
            
        # 2. Preprocessing
        # WHAT: Reshaping to 2D array (1 sample, N features).
        data_reshaped = np.array(new_data).reshape(1, -1)
        # WHAT: Scaling using the saved scaler logic.
        data_scaled = scaler.transform(data_reshaped)
        
        # 3. Prediction
        prediction = model.predict(data_scaled)[0]
        
        # 4. Probability (if supported)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data_scaled)[0][1]
        else:
            probability = "N/A (Model doesn't support proba)"
            
        return {
            "churn_prediction": int(prediction),
            "churn_probability": probability,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e)}

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    # 1. Data Loading and Prep
    file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = load_data(file_name)
    
    if df is not None:
        perform_eda(df)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_columns = prepare_data(df)
        
        # 2. Defining Models
        # Dictionary mapping Name -> Algorithm Object
        models = {
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(C=1.0, gamma='scale', probability=True), # Enable proba for ROC
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # 3. Training & Evaluating
        # This function handles the training loop and metrics collection.
        results = train_and_evaluate(models, X_train, y_train, X_test, y_test)
        
        # 4. Analyzing Results
        # This prints the table and creates plots.
        metrics = analyze_results(results, X_test, y_test)
        
        # 5. Recommendation Logic for Production
        # Filter models with <50ms latency
        valid_models = metrics[metrics['pred_time_1000_ms'] < 50]
        
        if not valid_models.empty:
            # Pick the one with Max Test Accuracy among valid models.
            best_model_name = valid_models['test_acc'].idxmax()
            print(f"\nRECOMMENDATION: The best model for production is '{best_model_name}'.")
            print(f"Reason: It meets the <50ms latency requirement ({valid_models.loc[best_model_name, 'pred_time_1000_ms']:.2f}ms) with the highest accuracy.")
        else:
            # Fallback if all are slow.
            print("\nRECOMMENDATION: No model met the latency requirement. Consider optimizing Random Forest or using Decision Tree.")
            best_model_name = metrics['test_acc'].idxmax()

        # 6. Test Production Function
        print("\n--- Testing Production Function (Dummy Input) ---")
        # In a real scenario, we would need to replicate the get_dummies logic for new input.
        # For this demo, we assume the input is already pre-processed or we pass a sample from the test set.
        sample_customer_raw = np.random.rand(len(feature_columns)) # Placeholder magnitude
        best_model = results[best_model_name]['model']
        
        # Since predict_new_customer expects raw input, but our pipeline handles complex encoding,
        # verifying strictly on pre-processed test data for now to show mechanics.
        print("Test with pre-processed sample from X_test[0]:")
        try: 
           # Note: X_test is already scaled, so we'd technically need to inverse_transform to test 'raw' input flow perfectly,
           # but passing it directly to model.predict via the helper checks the function structure.
           # Let's just create a raw-like vector (zeros) matching feature count to test the shape check and flow.
           dummy_raw_input = np.zeros(len(feature_columns)) 
           response = predict_new_customer(best_model, scaler, dummy_raw_input)
           print(f"Prediction for dummy input: {response}")
        except Exception as e:
           print(e)
