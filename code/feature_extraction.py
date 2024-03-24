import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from cnn_model import CNN  # Make sure this import matches your project structure

def extract_features(model, file_path):
    # Load the image, ensuring the target size and color mode match your training setup
    img = image.load_img(file_path, target_size=(128, 128), color_mode='grayscale')  # Adjust as necessary
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize

    # Assuming 'model' is an instance of your custom CNN and has been trained
    # Modify the model to output features from the dense layer before the dropout
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.fc_layer.output)

    # Extract features
    features = feature_extractor.predict(img_array)

    return features
