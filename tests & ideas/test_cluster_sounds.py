# import os
# import librosa
# import numpy as np
# import concurrent.futures
# from sklearn.cluster import DBSCAN
# import shutil

# def process_audio(file_path):
#     # Load the audio file with a lower sampling rate
#     y, sr = librosa.load(file_path, sr=22050)  # Downsample to 22050 Hz

#     # Extract fewer MFCC coefficients (e.g., 13 instead of the default 20)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

#     return mfcc

# def compute_similarity(mfcc1, mfcc2):
#     # Ensure both MFCC arrays have the same shape by padding
#     max_length = max(mfcc1.shape[1], mfcc2.shape[1])
#     mfcc1 = np.pad(mfcc1, ((0, 0), (0, max_length - mfcc1.shape[1])), mode='constant')
#     mfcc2 = np.pad(mfcc2, ((0, 0), (0, max_length - mfcc2.shape[1])), mode='constant')

#     # Compute the similarity (e.g., using cosine similarity)
#     similarity = np.dot(mfcc1.flatten(), mfcc2.flatten()) / (np.linalg.norm(mfcc1.flatten()) * np.linalg.norm(mfcc2.flatten()))

#     return similarity

# def cluster_audio_files(folder_path, output_folder, eps=0.5, min_samples=2):
#     # List all audio files in the folder
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

#     # Process audio files in parallel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         mfccs = list(executor.map(process_audio, audio_files))

#     # Compute the similarity matrix
#     similarity_matrix = np.zeros((len(audio_files), len(audio_files)))
#     for i in range(len(audio_files)):
#         for j in range(i, len(audio_files)):
#             similarity = compute_similarity(mfccs[i], mfccs[j])
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity

#     # Apply DBSCAN clustering
#     dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(1 - similarity_matrix)  # Use 1 - similarity to convert to distance

#     # Create output folders and move files
#     for label in set(labels):
#         cluster_folder = os.path.join(output_folder, f'cluster_{label}')
#         os.makedirs(cluster_folder, exist_ok=True)
#         for i, file in enumerate(audio_files):
#             if labels[i] == label:
#                 shutil.move(file, os.path.join(cluster_folder, os.path.basename(file)))

# # Example usage
# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# cluster_audio_files(folder_path, output_folder)

# import os
# import librosa
# import numpy as np
# import concurrent.futures
# from sklearn.cluster import DBSCAN
# import faiss  # Facebook AI Similarity Search
# import shutil

# def process_audio(file_path, max_len=100):
#     y, sr = librosa.load(file_path, sr=22050)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     if mfcc.shape[1] < max_len:
#         mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]
#     return mfcc

# def compute_similarity_matrix(mfccs):
#     d = mfccs[0].shape[0] * mfccs[0].shape[1]
#     index = faiss.IndexFlatL2(d)
#     mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
#     index.add(mfccs_flat)
#     D, I = index.search(mfccs_flat, k=len(mfccs))
#     return D

# def normalize_similarity_matrix(D):
#     D_min = D.min()
#     D_max = D.max()
#     D_normalized = (D - D_min) / (D_max - D_min)
#     return D_normalized

# def cluster_audio_files(folder_path, output_folder, eps=0.7, min_samples=2):
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         mfccs = list(executor.map(process_audio, audio_files))

#     similarity_matrix = compute_similarity_matrix(mfccs)
#     normalized_similarity_matrix = normalize_similarity_matrix(similarity_matrix)

#     dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(1 - normalized_similarity_matrix)

#     for label in set(labels):
#         cluster_folder = os.path.join(output_folder, f'cluster_{label}')
#         os.makedirs(cluster_folder, exist_ok=True)
#         for i, file in enumerate(audio_files):
#             if labels[i] == label:
#                 shutil.move(file, os.path.join(cluster_folder, os.path.basename(file)))

# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# cluster_audio_files(folder_path, output_folder)

# import os
# import librosa
# import numpy as np
# import concurrent.futures
# from sklearn.cluster import KMeans
# import faiss  # Facebook AI Similarity Search
# import shutil

# def process_audio(file_path, max_len=100):
#     y, sr = librosa.load(file_path, sr=22050)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     if mfcc.shape[1] < max_len:
#         mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]
#     return mfcc

# def compute_similarity_matrix(mfccs):
#     d = mfccs[0].shape[0] * mfccs[0].shape[1]
#     index = faiss.IndexFlatL2(d)
#     mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
#     index.add(mfccs_flat)
#     D, I = index.search(mfccs_flat, k=len(mfccs))
#     return D

# def normalize_similarity_matrix(D):
#     D_min = D.min()
#     D_max = D.max()
#     D_normalized = (D - D_min) / (D_max - D_min)
#     return D_normalized

# def cluster_audio_files(folder_path, output_folder, n_clusters=5):
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         mfccs = list(executor.map(process_audio, audio_files))

#     similarity_matrix = compute_similarity_matrix(mfccs)
#     normalized_similarity_matrix = normalize_similarity_matrix(similarity_matrix)

#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     labels = kmeans.fit_predict(normalized_similarity_matrix)

#     for label in set(labels):
#         cluster_folder = os.path.join(output_folder, f'cluster_{label}')
#         os.makedirs(cluster_folder, exist_ok=True)
#         for i, file in enumerate(audio_files):
#             if labels[i] == label:
#                 shutil.move(file, os.path.join(cluster_folder, os.path.basename(file)))

# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# cluster_audio_files(folder_path, output_folder)

# import os
# import librosa
# import numpy as np
# import concurrent.futures
# from sklearn.cluster import KMeans
# import faiss  # Facebook AI Similarity Search
# import shutil
# from difflib import SequenceMatcher

# def process_audio(file_path, max_len=100):
#     y, sr = librosa.load(file_path, sr=22050)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     if mfcc.shape[1] < max_len:
#         mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]
#     return mfcc

# def compute_similarity_matrix(mfccs):
#     d = mfccs[0].shape[0] * mfccs[0].shape[1]
#     index = faiss.IndexFlatL2(d)
#     mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
#     index.add(mfccs_flat)
#     D, I = index.search(mfccs_flat, k=len(mfccs))
#     return D

# def normalize_similarity_matrix(D):
#     D_min = D.min()
#     D_max = D.max()
#     if D_max == D_min:
#         D_normalized = np.zeros_like(D)
#     else:
#         D_normalized = (D - D_min) / (D_max - D_min)
#     return D_normalized

# def cluster_by_length(audio_files, n_clusters=5):
#     lengths = [librosa.get_duration(path=f) for f in audio_files]
#     lengths = np.array(lengths).reshape(-1, 1)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     labels = kmeans.fit_predict(lengths)
#     return labels

# def cluster_by_filename_similarity(audio_files, n_clusters=5):
#     def similarity(a, b):
#         return SequenceMatcher(None, a, b).ratio()
    
#     n = len(audio_files)
#     similarity_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             similarity_matrix[i, j] = similarity(audio_files[i], audio_files[j])
    
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     labels = kmeans.fit_predict(similarity_matrix)
#     return labels

# def cluster_audio_files(folder_path, output_folder, n_clusters=5):
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

#     # Cluster by length
#     length_labels = cluster_by_length(audio_files, n_clusters=n_clusters)
#     length_clusters = {i: [] for i in set(length_labels)}
#     for i, file in enumerate(audio_files):
#         length_clusters[length_labels[i]].append(file)

#     for length_label, length_cluster in length_clusters.items():
#         # Cluster by filename similarity within each length cluster
#         filename_labels = cluster_by_filename_similarity(length_cluster, n_clusters=n_clusters)
#         filename_clusters = {i: [] for i in set(filename_labels)}
#         for i, file in enumerate(length_cluster):
#             filename_clusters[filename_labels[i]].append(file)

#         for filename_label, filename_cluster in filename_clusters.items():
#             if len(filename_cluster) < n_clusters:
#                 n_clusters = len(filename_cluster)
#             # Cluster by audio similarity within each filename similarity cluster
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 mfccs = list(executor.map(process_audio, filename_cluster))

#             similarity_matrix = compute_similarity_matrix(mfccs)
#             normalized_similarity_matrix = normalize_similarity_matrix(similarity_matrix)

#             kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#             audio_labels = kmeans.fit_predict(normalized_similarity_matrix)

#             for audio_label in set(audio_labels):
#                 cluster_folder = os.path.join(output_folder, f'length_{length_label}', f'filename_{filename_label}', f'audio_{audio_label}')
#                 os.makedirs(cluster_folder, exist_ok=True)
#                 for i, file in enumerate(filename_cluster):
#                     if audio_labels[i] == audio_label:
#                         shutil.move(file, os.path.join(cluster_folder, os.path.basename(file)))

# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# cluster_audio_files(folder_path, output_folder)

# import os
# import librosa
# import numpy as np
# import concurrent.futures
# import shutil
# import faiss  # Facebook AI Similarity Search

# # Define the bins
# bins = ["clap", "snare", "kick", "hihat", "loops", "drum loops", "samples", "perc", "synth", "guitar"]

# def process_audio(file_path, max_len=100):
#     try:
#         y, sr = librosa.load(file_path, sr=22050)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(2048, len(y)), n_mels=40)
#         if mfcc.shape[1] < max_len:
#             mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
#         else:
#             mfcc = mfcc[:, :max_len]
#         return mfcc
#     except ValueError as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
#     except RuntimeError as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
#     except Exception as e:
#         print(f"Unexpected error processing {file_path}: {e}")
#         return None


# def compute_similarity_matrix(mfccs):
#     d = mfccs[0].shape[0] * mfccs[0].shape[1]
#     index = faiss.IndexFlatL2(d)
#     mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
#     index.add(mfccs_flat)
#     D, I = index.search(mfccs_flat, k=len(mfccs))
#     return D

# def sort_files_by_name(audio_files, bins):
#     sorted_files = {bin_name: [] for bin_name in bins}
#     remaining_files = []

#     for file in audio_files:
#         file_name = os.path.basename(file).lower()
#         matched_bins = [bin_name for bin_name in bins if bin_name in file_name]
#         if len(matched_bins) == 1:
#             sorted_files[matched_bins[0]].append(file)
#         else:
#             remaining_files.append(file)

#     return sorted_files, remaining_files

# def sort_remaining_files_by_similarity(remaining_files, sorted_files, bins, max_len=100):
#     bin_representatives = {}

#     for bin_name, files in sorted_files.items():
#         if files:
#             bin_representatives[bin_name] = process_audio(files[0], max_len=max_len)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         remaining_mfccs = list(executor.map(process_audio, remaining_files))

#     for i, mfcc in enumerate(remaining_mfccs):
#         max_similarity = -1
#         best_bin = None
#         for bin_name, bin_mfcc in bin_representatives.items():
#             similarity = np.dot(mfcc.flatten(), bin_mfcc.flatten()) / (np.linalg.norm(mfcc.flatten()) * np.linalg.norm(bin_mfcc.flatten()))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_bin = bin_name

#         if best_bin:
#             sorted_files[best_bin].append(remaining_files[i])

#     return sorted_files

# def organize_audio_files(folder_path, output_folder, bins):
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

#     # Step 1: Sort files by name
#     sorted_files, remaining_files = sort_files_by_name(audio_files, bins)

#     # Step 2: Sort remaining files by similarity
#     remaining_files = [file for file in remaining_files if process_audio(file) is not None]
#     sorted_files = sort_remaining_files_by_similarity(remaining_files, sorted_files, bins)

#     # Move files to respective folders
#     for bin_name, files in sorted_files.items():
#         bin_folder = os.path.join(output_folder, bin_name)
#         os.makedirs(bin_folder, exist_ok=True)
#         for file in files:
#             shutil.move(file, os.path.join(bin_folder, os.path.basename(file)))

# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# organize_audio_files(folder_path, output_folder, bins)

# import os
# import librosa
# import numpy as np
# import concurrent.futures
# import shutil
# import faiss  # Facebook AI Similarity Search

# # Define the bins
# loop_bins = ["loops", "drum loops", "samples", "top loops", "hihat loops"]
# one_shot_bins = ["clap", "crash", "ride", "snare", "kick", "hihat", "perc", "synth", "guitar"]

# def process_audio(file_path, max_len=100):
#     try:
#         y, sr = librosa.load(file_path, sr=22050)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(2048, len(y)), n_mels=40)
#         if mfcc.shape[1] < max_len:
#             mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
#         else:
#             mfcc = mfcc[:, :max_len]
#         return mfcc
#     except ValueError as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
#     except RuntimeError as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
#     except Exception as e:
#         print(f"Unexpected error processing {file_path}: {e}")
#         return None

# def compute_similarity_matrix(mfccs):
#     d = mfccs[0].shape[0] * mfccs[0].shape[1]
#     index = faiss.IndexFlatL2(d)
#     mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
#     index.add(mfccs_flat)
#     D, I = index.search(mfccs_flat, k=len(mfccs))
#     return D

# def sort_files_by_name(audio_files, bins):
#     sorted_files = {bin_name: [] for bin_name in bins}
#     remaining_files = []

#     for file in audio_files:
#         file_name = os.path.basename(file).lower()
#         matched_bins = [bin_name for bin_name in bins if bin_name in file_name]
#         if len(matched_bins) == 1:
#             sorted_files[matched_bins[0]].append(file)
#         else:
#             remaining_files.append(file)

#     return sorted_files, remaining_files

# def sort_remaining_files_by_similarity(remaining_files, sorted_files, bins, max_len=100):
#     bin_representatives = {}

#     for bin_name, files in sorted_files.items():
#         if files:
#             bin_representatives[bin_name] = process_audio(files[0], max_len=max_len)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         remaining_mfccs = list(executor.map(process_audio, remaining_files))

#     for i, mfcc in enumerate(remaining_mfccs):
#         max_similarity = -1
#         best_bin = None
#         for bin_name, bin_mfcc in bin_representatives.items():
#             similarity = np.dot(mfcc.flatten(), bin_mfcc.flatten()) / (np.linalg.norm(mfcc.flatten()) * np.linalg.norm(bin_mfcc.flatten()))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_bin = bin_name

#         if best_bin:
#             sorted_files[best_bin].append(remaining_files[i])

#     return sorted_files

# def organize_audio_files(folder_path, output_folder, bins):
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.aiff', '.flac'))]

#     # Step 1: Sort files by name
#     sorted_files, remaining_files = sort_files_by_name(audio_files, bins)

#     # Step 2: Sort remaining files by similarity
#     remaining_files = [file for file in remaining_files if process_audio(file) is not None]
#     sorted_files = sort_remaining_files_by_similarity(remaining_files, sorted_files, bins)

#     # Move files to respective folders
#     for bin_name, files in sorted_files.items():
#         bin_folder = os.path.join(output_folder, bin_name)
#         os.makedirs(bin_folder, exist_ok=True)
#         for file in files:
#             shutil.move(file, os.path.join(bin_folder, os.path.basename(file)))

# folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/_audio_files'
# output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
# organize_audio_files(folder_path, output_folder, bins)


import os
import librosa
import numpy as np
import concurrent.futures
import shutil
import faiss  # Facebook AI Similarity Search

# Define the bins
loop_bins = ["loops", "drum loops", "samples", "top loops", "hihat loops"]
one_shot_bins = ["clap", "crash", "ride", "snare", "kick", "hihat", "perc", "synth", "guitar"]

# Define the threshold for long and short samples (in seconds)
threshold_duration = 4.0

def process_audio(file_path, max_len=100):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(2048, len(y)), n_mels=40)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except ValueError as e:
        print(f"Error processing {file_path}: {e}")
        return None
    except RuntimeError as e:
        print(f"Error processing {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return None

def compute_similarity_matrix(mfccs):
    d = mfccs[0].shape[0] * mfccs[0].shape[1]
    index = faiss.IndexFlatL2(d)
    mfccs_flat = np.array([mfcc.flatten() for mfcc in mfccs])
    index.add(mfccs_flat)
    D, I = index.search(mfccs_flat, k=len(mfccs))
    return D

def sort_files_by_name(audio_files, bins):
    sorted_files = {bin_name: [] for bin_name in bins}
    remaining_files = []

    for file in audio_files:
        file_name = os.path.basename(file).lower()
        matched_bins = [bin_name for bin_name in bins if bin_name in file_name]
        if len(matched_bins) == 1:
            sorted_files[matched_bins[0]].append(file)
        else:
            remaining_files.append(file)

    return sorted_files, remaining_files

def sort_remaining_files_by_similarity(remaining_files, sorted_files, bins, max_len=100):
    bin_representatives = {}

    for bin_name, files in sorted_files.items():
        if files:
            bin_representatives[bin_name] = process_audio(files[0], max_len=max_len)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        remaining_mfccs = list(executor.map(process_audio, remaining_files))

    for i, mfcc in enumerate(remaining_mfccs):
        max_similarity = -1
        best_bin = None
        for bin_name, bin_mfcc in bin_representatives.items():
            similarity = np.dot(mfcc.flatten(), bin_mfcc.flatten()) / (np.linalg.norm(mfcc.flatten()) * np.linalg.norm(bin_mfcc.flatten()))
            if similarity > max_similarity:
                max_similarity = similarity
                best_bin = bin_name

        if best_bin:
            sorted_files[best_bin].append(remaining_files[i])

    return sorted_files

def organize_audio_files(folder_path, output_folder, loop_bins, one_shot_bins, threshold_duration=2.0):
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.aiff', '.flac'))]

    # Step 1: Cluster files into long and short samples
    long_samples = []
    short_samples = []

    for file in audio_files:
        try:
            y, sr = librosa.load(file, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration >= threshold_duration:
                long_samples.append(file)
            else:
                short_samples.append(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Step 2: Sort files by name within each group
    sorted_long_files, remaining_long_files = sort_files_by_name(long_samples, loop_bins)
    sorted_short_files, remaining_short_files = sort_files_by_name(short_samples, one_shot_bins)

    # Step 3: Sort remaining files by similarity within each group
    remaining_long_files = [file for file in remaining_long_files if process_audio(file) is not None]
    sorted_long_files = sort_remaining_files_by_similarity(remaining_long_files, sorted_long_files, loop_bins)

    remaining_short_files = [file for file in remaining_short_files if process_audio(file) is not None]
    sorted_short_files = sort_remaining_files_by_similarity(remaining_short_files, sorted_short_files, one_shot_bins)

    # Move files to respective folders
    for bin_name, files in sorted_long_files.items():
        bin_folder = os.path.join(output_folder, "long_samples", bin_name)
        os.makedirs(bin_folder, exist_ok=True)
        for file in files:
            shutil.move(file, os.path.join(bin_folder, os.path.basename(file)))

    for bin_name, files in sorted_short_files.items():
        bin_folder = os.path.join(output_folder, "short_samples", bin_name)
        os.makedirs(bin_folder, exist_ok=True)
        for file in files:
            shutil.move(file, os.path.join(bin_folder, os.path.basename(file)))

folder_path = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/_audio_files'
output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder compare/clusters'
organize_audio_files(folder_path, output_folder, loop_bins, one_shot_bins, threshold_duration)