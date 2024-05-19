from data_preprocessing import load_data
from cnn_model import CNN
from train import train_model
from feature_extraction import extract_features
from svm import train_svm, evaluate_svm, save_model, load_model
import numpy as np
import tensorflow as tf

def main():
    # Define your paths, parameters, and other configurations
    base_path = 'data/'
    input_shape = (256, 100, 1)  # New input shape after resizing
    num_classes = 10 
    batch_size = 16
    epochs = 20
    # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Step 1: Data Preprocessing
    print("Loading and preprocessing data...")
    train_generator, val_generator, test_generator = load_data(base_path, img_size=input_shape[:2], batch_size=batch_size)
    
    # Step 2: CNN Training
    print("Training CNN model...")
    cnn_model = CNN(num_classes=num_classes)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    
    # Step 3: Feature Extraction
    # Note: This step assumes you adapt your feature_extraction.py to work on batches or entire datasets.
    print("Extracting features using trained CNN...")
    train_features, train_labels = extract_features(cnn_model, train_generator)
    test_features, test_labels = extract_features(cnn_model, test_generator)
    
    # Step 4: SVM Training
    print("Training SVM classifier...")
    svm_classifier = train_svm(train_features, train_labels)
    
    # Optionally, save the trained SVM model
    save_model(svm_classifier, 'svm_classifier.joblib')
    
    # Step 5: Evaluation
    print("Evaluating SVM classifier...")
    evaluate_svm(svm_classifier, test_features, test_labels)
    
if __name__ == '__main__':
    main()
