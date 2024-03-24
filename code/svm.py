# svm.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib  # For saving the model

def train_svm(train_features, train_labels):
    """
    Trains an SVM classifier.
    
    Parameters:
    - train_features: numpy array of features for training data
    - train_labels: numpy array of labels for training data
    
    Returns:
    - trained SVM model
    """
    print("Training SVM...")
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(train_features, train_labels)
    print("SVM trained successfully.")
    return svm_classifier

def evaluate_svm(svm_classifier, test_features, test_labels):
    """
    Evaluates the SVM classifier on the test set.
    
    Parameters:
    - svm_classifier: trained SVM model
    - test_features: numpy array of features for test data
    - test_labels: numpy array of labels for test data
    """
    predictions = svm_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy}")
    print(classification_report(test_labels, predictions))

def save_model(svm_classifier, file_path='svm_classifier.joblib'):
    """
    Saves the trained SVM model to a file.
    
    Parameters:
    - svm_classifier: trained SVM model
    - file_path: path to save the model
    """
    joblib.dump(svm_classifier, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path='svm_classifier.joblib'):
    """
    Loads an SVM model from a file.
    
    Parameters:
    - file_path: path to the model file
    
    Returns:
    - Loaded SVM model
    """
    return joblib.load(file_path)
