import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_mel_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

audio_dir = ['Data/genres_original/country','Data/genres_original/disco', 'Data/genres_original/hiphop','Data/genres_original/jazz',
              'Data/genres_original/metal','Data/genres_original/pop', 'Data/genres_original/reggae', 'Data/genres_original/rock']
spectrogram_dir = ['Data/spectrograms/country','Data/spectrograms/disco', 'Data/spectrograms/hiphop','Data/spectrograms/jazz',
              'Data/spectrograms/metal','Data/spectrograms/pop', 'Data/spectrograms/reggae', 'Data/spectrograms/rock']


for audio, spectrogram in zip(audio_dir, spectrogram_dir):
    for filename in os.listdir(audio):
        if filename.endswith('.wav'):
            try:
                audio_path = os.path.join(audio, filename)
                save_path = os.path.join(spectrogram, filename.replace('.wav', '.png'))
                create_mel_spectrogram(audio_path, save_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

