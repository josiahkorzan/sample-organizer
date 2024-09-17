# import os
# import librosa
# import numpy as np
# import shutil

# # Load the audio files
# y1, sr1 = librosa.load('test folder compare/MaxV - Bongo2__Acetone Rhythm King.wav')
# y2, sr2 = librosa.load('test folder compare/MaxV - Bongo__Acetone Rhythm King.wav')

# # Extract features (e.g., Mel-frequency cepstral coefficients - MFCCs)
# mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
# mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)

# # Ensure both MFCC arrays have the same shape by padding
# max_length = max(mfcc1.shape[1], mfcc2.shape[1])
# mfcc1 = np.pad(mfcc1, ((0, 0), (0, max_length - mfcc1.shape[1])), mode='constant')
# mfcc2 = np.pad(mfcc2, ((0, 0), (0, max_length - mfcc2.shape[1])), mode='constant')

# # Compute the similarity (e.g., using cosine similarity)
# similarity = np.dot(mfcc1.flatten(), mfcc2.flatten()) / (np.linalg.norm(mfcc1.flatten()) * np.linalg.norm(mfcc2.flatten()))

# print(f'Similarity: {similarity}')

# folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder un-nested/_audio_files'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder un-nested/similar'

import os
import librosa
import numpy as np
import concurrent.futures

def process_audio(file_path):
    # Load the audio file with a lower sampling rate
    y, sr = librosa.load(file_path, sr=22050)  # Downsample to 22050 Hz

    # Extract fewer MFCC coefficients (e.g., 13 instead of the default 20)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return mfcc

def compute_similarity(mfcc1, mfcc2):
    # Ensure both MFCC arrays have the same shape by padding
    max_length = max(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = np.pad(mfcc1, ((0, 0), (0, max_length - mfcc1.shape[1])), mode='constant')
    mfcc2 = np.pad(mfcc2, ((0, 0), (0, max_length - mfcc2.shape[1])), mode='constant')

    # Compute the similarity (e.g., using cosine similarity)
    similarity = np.dot(mfcc1.flatten(), mfcc2.flatten()) / (np.linalg.norm(mfcc1.flatten()) * np.linalg.norm(mfcc2.flatten()))

    return similarity

# List of audio files to process
audio_files = ['test folder compare/MaxV - Bongo2__Acetone Rhythm King.wav', 'test folder compare/MaxV - Bongo__Acetone Rhythm King.wav']

# Process audio files in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    mfccs = list(executor.map(process_audio, audio_files))

# Compute similarity between the first two audio files
similarity = compute_similarity(mfccs[0], mfccs[1])

print(f'Similarity: {similarity}')

