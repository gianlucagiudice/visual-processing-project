from config import FULL_UTK_CROPPED_PATH, FULL_UTK_CROPPED_METADATA_FILENAME
from src.DataManager import DataManager
from src.models.HandcraftedModel import HandcraftedModel
from src.models.Model import IMAGE_INPUT_SIZE

N_SAMPLE = 1

# Read the data
data_manager = DataManager(FULL_UTK_CROPPED_PATH, FULL_UTK_CROPPED_METADATA_FILENAME, IMAGE_INPUT_SIZE,
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
       'ACCCLF;RECCLF;FSCORECLF;MSEREG;RMSEREG;MAEREG;TOP5ACC;TOP10ACC;TOP15ACC;TOP20ACC;TIME'
f.write(line + '\n')
f.close()

# TODO: prendi modello migliore e ritorna tempi di predizione per ogni img in un nuovo file txt

# Define the model
# n_sift = 25  # explain with velocity
# color_hist_bins = 128
# lbp_n_points = 24  # little increase of performance with these values
# lbp_radius = 3
compute_sift = True
compute_hog = True
compute_hist = True
compute_lbp = True
# handcrafted_model = HandcraftedModel(data_manager, n_sift, color_hist_bins, lbp_n_points, lbp_radius, compute_sift,
#                                      compute_hog, compute_hist, compute_lbp)
# Train model
# handcrafted_model.train(X_train, y_train, X_test, y_test, X_val, y_val)

# number of best sift descriptors
n_sift_range = [10, 25, 50]
# bins of the color histogram
color_hist_bins_range = [32, 64, 128]
# values for lbp
lbp_values_range = [[8, 1], [16, 2], [24, 3]]

for n_sift in n_sift_range:
    for color_hist_bins in color_hist_bins_range:
        for lbp_values in lbp_values_range:
            print('1 of ' + str(len(n_sift_range)*len(color_hist_bins_range)*len(lbp_values_range)) + ' steps')
            handcrafted_model = HandcraftedModel(data_manager, n_sift, color_hist_bins, lbp_values[0], lbp_values[1],
                                                 compute_sift, compute_hog, compute_hist, compute_lbp)
            handcrafted_model.train(X_train, y_train, X_test, y_test, X_val, y_val)

