from config import UTK_PATH, UTK_METADATA_FILENAME
from src.DataManager import DataManager
from src.models.HandcraftedModel import HandcraftedModel
from src.models.Model import IMAGE_INPUT_SIZE

N_SAMPLE = 0.1

# Read the data
data_manager = DataManager(UTK_PATH, UTK_METADATA_FILENAME, IMAGE_INPUT_SIZE,
                           n_subset=N_SAMPLE, normalize_images=True, normalize_age=True)
data = data_manager.get_dataset()

# Split into train, validation, test
train, validation, test = data_manager.split_dataset(data)

# Read images
print('Read training images ...')
X_train, y_train = data_manager.get_X(train), data_manager.get_y(train)
print('Read validation images ...')
X_val, y_val = data_manager.get_X(validation), data_manager.get_y(validation)
print('Read test images ...')
X_test, y_test = data_manager.get_X(test), data_manager.get_y(test)

f = open('evaluation_handcrafted.txt', 'a')
line = 'HC;NSIFT;HISTBINS;LBPPOINTS;LBPRADIUS;COMPUTESIFT;COMPUTEHOG;COMPUTEHIST;COMPUTELBP;' + \
       'ACCCLF;MSEREG;RMSEREG;MAEREG;TIME'
f.write(line + '\n')

# Define the model
print('SIFT')
n_sift = 50
color_hist_bins = 128
lbp_n_points = 24
lbp_radius = 3
compute_sift = True
compute_hog = False
compute_hist = False
compute_lbp = False
handcrafted_model = HandcraftedModel(data_manager, n_sift, color_hist_bins, lbp_n_points, lbp_radius, compute_sift,
                                     compute_hog, compute_hist, compute_lbp)

# Train model
handcrafted_model.train(X_train, y_train, X_test, y_test, X_val, y_val)
