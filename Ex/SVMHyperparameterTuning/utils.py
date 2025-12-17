"""
utils.py
Shared utility functions for SVM Hyperparameter Tuning.
Handles data loading (with fallback), preprocessing, and common evaluation metrics.
"""

# ----------------- IMPORTS -----------------

# WHAT: Standard libraries for data and regex.
# WHY: 're' is needed for text cleaning (removing special chars).
# EXPECTED OUTPUT: Modules available.
import pandas as pd
import numpy as np
import re
import time
import os
import ssl

# Bypass SSL verification for legacy/proxy environments
ssl._create_default_https_context = ssl._create_unverified_context

# WHAT: Scikit-learn libraries.
# WHY: TF-IDF for text features, metrics for evaluation, datasets for fallback.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_20newsgroups

# ----------------- DATA LOADING & PREPROCESSING -----------------

def load_and_preprocess_data():
    """
    Loads dataset (SMS Spam OR 20 Newsgroups), preprocesses text, and returns TF-IDF vectors.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("--- Data Loading Phase ---")
    
    # WHAT: Check for SMS Spam Collection formatted file.
    # Note: Kaggle dataset usually is 'spam.csv' or 'SMSSpamCollection'.
    # We will look for a common name or fallback to 20newsgroups.
    file_path = 'SMSSpamCollection' # Expected raw file name if downloaded manually
    
    data = []
    labels = []
    
    # FLAGGING: Fallback logic.
    if os.path.exists(file_path):
        print(f"Found {file_path}. Loading SMS Spam Dataset...")
        # SMS Spam Collection is usually tab-separated: "ham \t message"
        try:
            df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
            # Convert label 'ham'/'spam' to 0/1.
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            data = df['message'].values
            labels = df['label'].values
        except Exception as e:
            print(f"Error reading SMS file: {e}")
            return None
    elif os.path.exists('spam.csv'):
         print("Found spam.csv. Loading...")
         try:
            df = pd.read_csv('spam.csv', encoding='latin-1')
            # Common Kaggle format: v1=label, v2=message
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            data = df['message'].values
            labels = df['label'].values
         except Exception as e:
            print(f"Error reading spam.csv: {e}")
            return None
    else:
        print("SMS Datasets not found. Using Fallback: 20 Newsgroups (sci.crypt vs rec.autos).")
        # WHAT: Loading 2 categories from 20newsgroups to simulate binary classification.
        # WHY: To ensure the code runs even if the user hasn't downloaded the specific CSV.
        categories = ['sci.crypt', 'rec.autos']
        newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
        data = newsgroups.data
        labels = newsgroups.target # 0 and 1
        print(f"Loaded {len(data)} documents from 20 Newsgroups.")

    # WHAT: Text Preprocessing.
    # WHY: To clean noise (numbers, special chars) before vectorization.
    print("Preprocessing text...")
    clean_data = []
    for text in data:
        # 1. Lowercase
        text = text.lower()
        # 2. Remove special chars and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        clean_data.append(text)
        
    # WHAT: TF-IDF Vectorization.
    # METHOD: TfidfVectorizer(max_features=1000)
    # ARGUMENTS: max_features=1000. Limits vocab to top 1000 frequent distinctive words.
    # WHY: Reduces dimensionality (speed) and avoids overfitting on rare words.
    print("Vectorizing (TF-IDF, max_features=1000)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(clean_data)
    y = np.array(labels)
    
    # WHAT: Splitting Data (70% Train, 15% Val, 15% Test).
    # Step 1: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # Step 2: Split Temp into 50/50 (15% total each)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"Data Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------- EVALUATION HELPER -----------------

def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Evaluates a model and returns dictionary of metrics.
    """
    y_pred = model.predict(X)
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0)
    }

if __name__ == "__main__":
    # Test the loader
    load_and_preprocess_data()
