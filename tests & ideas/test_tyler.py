import wav_clustering_workflow_timed as wcw
import glob
import os

# Set the parent directory for your samples
parent_dir = '/Users/josiahkorzan/Desktop/Programming/Projects/Sample Organization Project/test folders/test folder output3/loops'

# Make a list of absolute paths to all the .wav files you want to analyze
pattern = os.path.join(parent_dir, '*.wav')
wavs = glob.glob(pattern)

# Now we are ready to call the main function
wcw.cluster_and_save_order(wavs, 2, parent_dir=parent_dir, outdir='output')