import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_path, img_size=(1000, 400), batch_size=32):
    """
    Loads data from directories and prepares it for training.

    Parameters:
    - base_path: The base directory containing 'train', 'validate', and 'test' folders.
    - img_size: Tuple specifying the image size. Default is (1000, 400) for Mel spectrograms.
    - batch_size: Batch size for the data generators.

    Returns:
    - train_generator, val_generator, test_generator: Data generators for training, validation, and testing.
    """
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(base_path, 'train'),
        target_size=(256, 100),  
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        directory=os.path.join(base_path, 'validation'),
        target_size=(256, 100),  
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(base_path, 'test'),
        target_size=(256, 100),  
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator
