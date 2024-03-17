import numpy as np
import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_data(base_path, genres, test_size=0.2, val_size=0.25):
    file_paths = []
    labels = []
    
    for genre in genres:
        genre_path = os.path.join(base_path, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.png'):
                file_paths.append(os.path.join(genre_path, filename))
                labels.append(genre)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
