from src.models.ModelFromScratch import ModelFromScratch
from src.DataManager import DataManager
from config import FULL_UTK_CROPPED_PATH, FULL_UTK_CROPPED_METADATA_FILENAME

INPUT_SIZE = (124, 124, 3)

N_SAMPLE = 1

# Read the data
data_manager = DataManager(FULL_UTK_CROPPED_PATH,
                           FULL_UTK_CROPPED_METADATA_FILENAME,
                           INPUT_SIZE,
                           test_size=0.2, validation_size=0.15,
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

model_from_scratch = ModelFromScratch(INPUT_SIZE)
model_from_scratch.train(X_train, y_train, X_val, y_val, X_test, y_test)



