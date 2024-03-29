from os.path import join
import random
import numpy as np

# Dataset path - imdb
IMDB_CROPPED_PATH = '../dataset/imdb_crop'
IMDB_CROPPED_METADATA_FILENAME = 'imdb.pickle'
IMDB_FAMOUS_ACTORS_FILENAME = 'imdb_most_famous_actors.pickle'
# utk
UTK_PATH = '../dataset/utk/images'
UTK_METADATA_FILENAME = 'utk_full.pickle'
# wiki
WIKI_PATH = '../dataset/imdb_crop'
WIKI_FAMOUS_ACTORS_FILENAME = 'wiki_most_famous_actors.pickle'

UTK_CROPPED_PATH = '../dataset/utk/images'
UTK_CROPPED_METADATA_FILENAME = '../utk_full.pickle'

FULL_UTK_CROPPED_PATH = '../dataset/utk/images'
FULL_UTK_CROPPED_METADATA_FILENAME = '../utk_full.pickle'

# Image folder
OUTPUT_IMAGE_FOLDER = join('..', 'doc', 'images')

# Report Folder
OUTPUT_REPORT_FOLDER = join('..', 'doc', 'reports')

# Checkpoint dir
CHECKPOINT_DIR = join('..', 'temp', 'checkpoint')

# Log dir
LOG_DIR = join('..', 'temp', 'log')

# Save model dir
SAVE_MODEL_DIR = join('..', 'model')

# Telegram Token
TELEGRAM_BOT_TOKEN = '5279649456:AAHnpo2sa8cRfh78RzAXTuvFTFAY4vZVbmU'

# Random seed
SEED = 830694
random.seed(SEED)
np.random.seed(SEED)
