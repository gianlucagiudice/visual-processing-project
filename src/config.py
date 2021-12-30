from os.path import join
import random
import numpy as np

# Dataset path
IMDB_CROPPED_PATH = '../dataset/imdb_crop'
IMBD_CROPPED_METADATA_FILENAME = 'imdb.pickle'

# Image folder
OUTPUT_IMAGE_FOLDER = join('..', 'doc', 'images')

# Report Folder
OUTPUT_REPORT_FOLDER = join('..', 'doc', 'reports')

# Checkpoint dir
CHECKPOINT_DIR = join('..', 'checkpoint')

# Log dir
LOG_DIR = join('..', 'log')



# Random seed
SEED = 830694
random.seed(SEED)
np.random.seed(SEED)
