import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def extract_features(model, file_path):
    # Load the image file, ensuring the target size matches the model's input
    img = image.load_img(file_path, target_size=(128, 128), color_mode='grayscale')  # Adjust target_size as necessary
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize

    # Assuming model is already compiled and trained, directly use it for prediction
    # Extract features from the pre-final layer
    # Here, we create a new Model that outputs features from a specific layer
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # Get the features
    features = intermediate_layer_model.predict(img_array)

    return features
# The `extract_features` function takes a trained model and a file path to a spectrogram image. It loads the image, preprocesses it, and extracts features from the pre-final layer of the model. The features are then returned as a NumPy array.