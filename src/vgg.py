from src.models.PretrainedVGG import PretrainedVGG
from src.DataManager import DataManager
from config import IMBD_CROPPED_METADATA_FILENAME, IMDB_CROPPED_PATH
from src.models.Model import IMAGE_INPUT_SIZE

N_SAMPLE = 700 # Con 70 funziona, con 71 no

# Read the data
data_manager = DataManager(IMDB_CROPPED_PATH, IMBD_CROPPED_METADATA_FILENAME, IMAGE_INPUT_SIZE,
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

pretrained_vgg16 = PretrainedVGG()
pretrained_vgg16.train(X_train, y_train, X_val, y_val, X_test, y_test)