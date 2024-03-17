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
