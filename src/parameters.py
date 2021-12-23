from os.path import join
import random
import numpy as np

# Dataset path
IMDB_CROPPED_PATH = '../dataset/imdb_crop'
IMBD_CROPPED_METADATA_FILENAME = 'imdb.pickle'

# Random seed
SEED = 830694
random.seed(SEED)
np.random.seed(SEED)
