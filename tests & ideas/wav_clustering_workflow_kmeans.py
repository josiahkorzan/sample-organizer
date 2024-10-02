import os
import glob
import pandas as pd
from shutil import copy
import pickle

from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
from sklearn.cluster import KMeans
import numpy as np

def get_short_term_features(wav_loc, win=0.050, step=0.050):
    """ 
    Extract short-term features using default 50msec non-overlapping windows
    """
    try:
        fs, s = aIO.read_audio_file(wav_loc)
        s = aIO.stereo_to_mono(s)
        [f, fn] = aF.feature_extraction(s, fs, int(fs * win), int(fs * step))
        return f.T  # Transpose to get features as rows
    except:
        return None

def flatten_n_frames(f, n):
    m = f[:, :n]
    return m.flatten('F')

def get_features_frame(wav_locs, first_n_frames, include_parent_dir=False):
    """
    Iterates over the list of paths to each wav file of interest. Extracts feature matrix. Subsets the feature matrix to the first_n_frames.
    """
    feature_dict = {}
    for w in wav_locs:
        wav_basename = w.split('/')[-1]
        if include_parent_dir:
            wav_dirname = w.split('/')[-2]
        try:
            f = get_short_term_features(w)
            feature_dict[wav_basename] = flatten_n_frames(f, first_n_frames)
        except TypeError:
            print(f'{w} appears to be too short to extract features')
    features_wavs_df = pd.DataFrame.from_dict(feature_dict, orient='index').transpose()
    return features_wavs_df

def save_ordered_wav_copies(parent_dir, list_of_dir_wavs, outdir):
    """ 
    Saves list of original filenames as a text file.
    Copies the files to a new, sorted location using the function copy() from shutil.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(os.path.join(outdir, "original_filenames.txt"), "w") as output:
        for orig_wav in list_of_dir_wavs:
            output.write(str(os.path.join(parent_dir, orig_wav)) + '\n')
    for i, source_wav in enumerate(list_of_dir_wavs):
        out_filename = str(i).zfill(5) + '_' + source_wav.replace('/', '_')
        copy(os.path.join(parent_dir, source_wav), os.path.join(outdir, out_filename))

def pickle_object(obj, outdir, out_name):
    with open(os.path.join(outdir, out_name), 'wb') as f:
        pickle.dump(obj, f)

def cluster_and_save_order(globbed_wav_list, n_frames, parent_dir, outdir):
    """
    Hierarchically clusters wav files and saves them with renamed files, sorted by similarity.
    """
    n_samples_in_glob = len(globbed_wav_list)
    if n_samples_in_glob < 3:
        print(f'{n_samples_in_glob} detected in input list: {globbed_wav_list}. Not enough to cluster.')
        raise FileNotFoundError
    df_matrix = get_features_frame(globbed_wav_list, n_frames, include_parent_dir=True)
    df_matrix.fillna(0, inplace=True)
    feature_matrix = df_matrix.T.values
    kmeans = KMeans(n_clusters=10).fit(feature_matrix)
    labels = kmeans.labels_
    sorted_indices = np.argsort(labels)
    drumnames = df_matrix.columns[sorted_indices]
    save_ordered_wav_copies(parent_dir, list_of_dir_wavs=drumnames, outdir=outdir)
    pickle_object(kmeans, outdir, 'kmeans_model.pkl')
    pickle_object(df_matrix, outdir, 'df_matrix.pkl')
    df_matrix.to_csv(os.path.join(outdir, 'all_features.csv'), index=False)
