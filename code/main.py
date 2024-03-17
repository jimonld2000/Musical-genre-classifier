from train import train_model
from feature_extraction import extract_features
import numpy as np

# Specify the base path to your dataset and define the genres
base_path = 'data/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Train the CNN model
# Note: You'll need to adjust the `input_shape` based on your spectrogram sizes
input_shape = (128, 128, 1)  # Height, Width, Channels
model = train_model(base_path, input_shape, batch_size=32, epochs=10)


# Example of extracting features from a new spectrogram
# Replace 'path/to/spectrogram.png' with the actual path to a spectrogram image
example_spectrogram_path = 'data/Data/spectrograms/blues/blues.00000.png'
features = extract_features(model, example_spectrogram_path)

print("Extracted features shape:", features.shape)
