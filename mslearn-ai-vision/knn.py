import numpy as np  # Used for numerical operations, though sklearn handles most here.
from sklearn.datasets import make_classification  # Function to generate synthetic (fake) data for testing.
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets.
from sklearn.neighbors import KNeighborsClassifier  # The k-Nearest Neighbors algorithm.
from sklearn.svm import SVC  # Support Vector Classification (SVM).
from sklearn.tree import DecisionTreeClassifier  # The Decision Tree algorithm.
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # Metrics to evaluate performance.
import time  # Used to measure how long the training takes.

def run_spam_classifier_comparison():
    # 1. Create Synthetic Dataset
    # make_classification creates a random dataset useful for testing algorithms.
    print("Generating synthetic dataset...")
    X, y = make_classification(
        # n_samples: Total number of rows (emails) to generate.
        # Example: n_samples=100 would generate a tiny dataset.
        n_samples=5000,
        
        # n_features: Total columns (data points per email like word counts, length).
        # Example: n_features=2 might be just "length" and "URL count".
        n_features=20,
        
        # n_informative: The number of features that actually help predict the class.
        # Example: If 15, then 15 columns contain useful signal, others are noise/redundant.
        n_informative=15, 
        
        # n_redundant: Features generated as random linear combinations of the informative features.
        # These trick the model; e.g., if Feature A is "word count", Feature B might be "word count * 2".
        n_redundant=2,
        
        # weights: The proportion of samples assigned to each class (0 and 1).
        # [0.9, 0.1] means 90% class 0 (Non-Spam) and 10% class 1 (Spam).
        # Example: weights=[0.5, 0.5] would be a perfectly balanced dataset.
        weights=[0.9, 0.1],
        
        # random_state: Seed for the random number generator.
        # Setting this ensures we get the EXACT same data every time we run the script.
        # Example: random_state=None would generate different data every run.
        random_state=42
    )

    # train_test_split divides the data into two parts: one to teach the model, one to test it.
    # X: The features (input data). y: The labels (answers).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        
        # test_size: The proportion of the dataset to include in the test split.
        # 0.2 means 20% of data is for testing, 80% for training.
        # Example: test_size=0.5 would split data in half.
        test_size=0.2, 
        
        # random_state=42: Ensures the split is the same every time we run.
        random_state=42, 
        
        # stratify=y: Crucial for imbalanced data!
        # It ensures the Train and Test sets have the same percentage of Spam vs Non-Spam as the original.
        # Without this, the test set might end up with 0 spam emails by chance.
        stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")  # shape[0] is the number of rows
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Spam count in test set: {sum(y_test)}\n") # sum(y_test) counts the 1s (Spambots)

    # 2. Define the Models dictionary
    models = {
        # KNeighborsClassifier: Looks at the 'k' closest points to decide the class.
        # n_neighbors=5: It looks at the 5 nearest neighbors.
        # Example: n_neighbors=1 is very sensitive to noise; n_neighbors=100 is very smooth/general.
        "k-Nearest Neighbors (k-NN)": KNeighborsClassifier(n_neighbors=5),
        
        # SVC: Support Vector Classifier.
        # kernel='linear': Uses a straight line/plane to separate data. Good for text.
        # Example: kernel='rbf' allows curved boundaries but is slower.
        # class_weight='balanced': Automatically adjusts weights inversely proportional to class frequencies.
        # This makes the model pay more attention to the minority class (Spam) so it doesn't ignore it.
        "Support Vector Machine (SVM)": SVC(kernel='linear', class_weight='balanced', random_state=42),
        
        # DecisionTreeClassifier: Splits data into branches/tree structures.
        # random_state=42: Trees can be random in how they select features; this fixes that randomness.
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # 3. Train and Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()  # Capture current time to measure speed.
        
        # fit(X, y): This is the instruction to "learn".
        # It takes the training inputs (X_train) and answers (y_train) and learns patterns.
        model.fit(X_train, y_train)
        
        # predict(X): Uses the learned patterns to guess labels for new, unseen data (X_test).
        y_pred = model.predict(X_test)
        
        elapsed = time.time() - start_time  # Calculate how many seconds passed.
        
        # Output Metrics
        print(f"\n--- Results for {name} ---")
        print(f"Time Elapsed: {elapsed:.4f}s")
        
        # accuracy_score: Returns the fraction of correctly classified samples.
        # (Correct Predictions) / (Total Predictions).
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # confusion_matrix: A table showing correct vs incorrect predictions.
        # [[True Negatives, False Positives],
        #  [False Negatives, True Positives]]
        # Shows exactly where the model is confused.
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # classification_report: Builds a text report showing main classification metrics.
        # Precision: Of all predicted spam, how many were actually spam?
        # Recall: Of all actual spam, how many did we catch?
        # F1-Score: A balance between Precision and Recall.
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))
        print("="*60 + "\n")

if __name__ == "__main__":
    run_spam_classifier_comparison()
