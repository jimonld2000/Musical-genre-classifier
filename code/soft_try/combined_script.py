# Combined Python Script

# Imports
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

# Data Preprocessing
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


# Feature Extraction
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def extract_features(model, file_path):
    # Load the image, ensuring the target size and color mode match your training setup
    img = image.load_img(file_path, target_size=(128, 128), color_mode='grayscale')  # Adjust as necessary
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize

    # Assuming you want to extract features directly with the trained model
    features = model.predict(img_array)

    return features


# Model Definition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(32, activation='relu'),
        Dropout(0.5),

        Dense(16, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Training
from data_preprocessing import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from code.cnn_model import create_cnn_model
import tensorflow as tf

def train_model(base_path, input_shape, batch_size=50, epochs=20):
    # Initialize the ImageDataGenerator with preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Assuming the structure of your data is base_path/train for training data
    # and base_path/validation for validation data
    train_generator = train_datagen.flow_from_directory(
        directory=base_path + 'train/',  # Adjust as necessary
        target_size=input_shape[:2],  # Height and width of the input images
        batch_size=batch_size,
        color_mode='grayscale',  # or 'rgb' if your images are in color
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        directory=base_path + 'validation/',  # Adjust as necessary
        target_size=input_shape[:2],  # Height and width of the input images
        batch_size=batch_size,
        color_mode='grayscale',  # or 'rgb' if your images are in color
        class_mode='categorical'
    )

    # Create the CNN model
    model = create_cnn_model(input_shape, len(train_generator.class_indices))

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    return model



# Main Workflow
from train import train_model
from feature_extraction import extract_features
import numpy as np

# Specify the base path to your dataset and define the genres
base_path = 'data/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Train the CNN model
# Note: You'll need to adjust the `input_shape` based on your spectrogram sizes
input_shape = (128, 128, 1)  # Height, Width, Channels
model = train_model(base_path, input_shape, batch_size=32, epochs=100)


# Example of extracting features from a new spectrogram
# Replace 'path/to/spectrogram.png' with the actual path to a spectrogram image
example_spectrogram_path = 'data/Data/spectrograms/blues/blues.00000.png'
# Ensure example_spectrogram_path is defined and points to a valid spectrogram image file
features = extract_features(model, example_spectrogram_path)
print("Extracted features:", features.shape)

