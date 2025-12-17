"""
Part7_Production.py
-------------------------------------------------------------------------------
Part 7: Production Deployment Pipeline

PROBLEM STATEMENT:
Once satisfied with the model, we need to "Deploy" it.
This means creating a standalone System/Class that can:
1. Load the trained model from disk.
2. Accept a raw transaction dictionary.
3. Preprocess it (using saved Scaler).
4. Return a fraud alert in milliseconds.

STEPS TO SOLVE:
1. Define `FraudDetectionSystem` class.
2. Implement `__init__` to load pickle files.
3. Implement `predict_fraud` for single-record inference.
4. Implement `train_and_save` to generate the pickle files.

CONCEPTS:
1. Pickle: Python's format for freezing objects (Model, Scaler) to files.
2. Scaler Persistence: We MUST use the exact same Mean/Std from training for production.
-------------------------------------------------------------------------------
"""

import pickle
import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Part2_DataPrep import prepare_data

# Constants
MODEL_PATH = 'fraud_model.pkl'
SCALER_PATH = 'scaler.pkl'

class FraudDetectionSystem:
    def __init__(self, model_path, scaler_path, threshold=0.5):
        """
        System Constructor. Loads artifacts into memory on startup.
        """
        try:
            # WHAT: Load Model from disk.
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            # WHAT: Load Scaler from disk.
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            # Store threshold config
            self.threshold = threshold
            print("Production System Initialized Successfully.")
        except FileNotFoundError:
            print("Error: Artifacts not found. Please train model first.")

    def predict_fraud(self, transaction_features):
        """
        Predicts if a single real-time transaction is fraud.
        
        ARGUMENTS:
        - transaction_features: list or dict of 30 feature values.
        
        RETURNS:
        - Dictionary with decision details.
        """
        start = time.time()
        
        # 1. format Input
        # WHAT: Convert dict to 1-row DataFrame.
        # WHY: Sklearn expects array-like structure.
        if isinstance(transaction_features, dict):
            df = pd.DataFrame([transaction_features])
        else:
            cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            df = pd.DataFrame([transaction_features], columns=cols)
            
        # 2. Preprocessing (Inference Time)
        # CRITICAL: Use self.scaler.transform(), NOT fit_transform().
        # We transform this one transaction using the statistics from the training set.
        df['Amount'] = self.scaler.transform(df[['Amount']])
        df['Time'] = self.scaler.transform(df[['Time']])
        
        # 3. Prediction
        # WHAT: Get probability of Class 1 (Fraud).
        prob = self.model.predict_proba(df)[:, 1][0]
        
        # 4. Decision
        is_fraud = prob >= self.threshold
        
        latency = (time.time() - start) * 1000 # Convert to ms
        
        return {
            'is_fraud': bool(is_fraud),
            'probability': float(prob),
            'latency_ms': latency,
            'alert': "FRAUD DETECTED" if is_fraud else "Normal"
        }

def train_and_save_production_model():
    """
    Simulates the End-To-End deployment process.
    """
    print("\n=== Part 7: Production Training & Deployment ===")
    
    # 1. Train
    print("Training Final Model (Balanced RF) on full data...")
    data = prepare_data()
    X_train, _, _, y_train, _, _, scaler = data
    
    # Train robust model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 2. Save
    print(f"Saving artifacts to {MODEL_PATH} and {SCALER_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # 3. Test
    print("Testing Live System...")
    # Initialize with tuned threshold (e.g. 0.3)
    system = FraudDetectionSystem(MODEL_PATH, SCALER_PATH, threshold=0.3)
    
    # Create Mock Input (Reverse Scaling one training sample)
    sample = X_train.iloc[0].to_dict()
    sample['Amount'] = scaler.inverse_transform([[sample['Amount']]])[0][0]
    sample['Time'] = scaler.inverse_transform([[sample['Time']]])[0][0]
    
    print("Processing mock transaction...")
    result = system.predict_fraud(sample)
    print("Prediction Result:", result)

if __name__ == "__main__":
    train_and_save_production_model()
