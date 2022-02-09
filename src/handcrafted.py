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

print('only sift')
handcrafted_model = HandcraftedModel(data_manager, 10, compute_sift=1, compute_hist=0, compute_hog=0, compute_lbp=0)
handcrafted_model.train(X_train, y_train, X_test, y_test, X_val, y_val)
