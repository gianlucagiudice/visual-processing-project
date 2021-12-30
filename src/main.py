from config import IMBD_CROPPED_METADATA_FILENAME, IMDB_CROPPED_PATH

from src.DataManager import DataManager
from src.models.Model import IMAGE_INPUT_SIZE

import matplotlib.pyplot as plt


N_SAMPLE = 25e3
N_SAMPLE = 10

# Read the data
data_manager = DataManager(IMDB_CROPPED_PATH, IMBD_CROPPED_METADATA_FILENAME, IMAGE_INPUT_SIZE, n_subset=N_SAMPLE)
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

# Train the model

plt.imshow(X_train[0])
plt.show()

plt.imshow(X_test[0])
plt.show()


plt.imshow(X_val[0])
plt.show()
pass