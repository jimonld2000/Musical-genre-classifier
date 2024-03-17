import os
import shutil
from sklearn.model_selection import train_test_split

# Base paths
src_base_path = '../../data/Data/spectrograms/'
dest_base_path = '../../data/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Split ratios
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

for genre in genres:
    src_genre_path = os.path.join(src_base_path, genre)
    files = [file for file in os.listdir(src_genre_path) if file.endswith('.png')]

    # Split files
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)
    validation_size = validation_ratio / (1 - test_ratio)
    train_files, val_files = train_test_split(train_files, test_size=validation_size, random_state=42)

    # Function to copy files to their new destination
    def copy_files(files, src_folder, dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
        for file in files:
            src_file_path = os.path.join(src_folder, file)
            dest_file_path = os.path.join(dest_folder, file)
            shutil.copy(src_file_path, dest_file_path)

    # Copy files to new directories
    for files, subset in zip([train_files, val_files, test_files], ['train', 'validation', 'test']):
        dest_path = os.path.join(dest_base_path, subset, genre)
        copy_files(files, src_genre_path, dest_path)

print("Data preparation complete.")
