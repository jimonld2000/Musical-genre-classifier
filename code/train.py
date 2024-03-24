from data_preprocessing import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from cnn_model import CNN  # Adjust this import based on your actual file structure

def train_model(base_path, input_shape, num_classes, batch_size=50, epochs=20):
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

    # Load data using the load_data function instead of flow_from_directory
    # This change assumes load_data returns data generators
    train_generator, val_generator, _, _ = load_data(base_path, genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
, batch_size=batch_size)  # Update genres list as necessary

    # Instantiate the CNN model
    model = CNN(num_classes=num_classes)

    # Compile the model (necessary if using custom training loops or the fit method without a compiled model)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    return model
