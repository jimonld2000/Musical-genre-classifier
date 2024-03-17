from data_preprocessing import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_cnn_model
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

