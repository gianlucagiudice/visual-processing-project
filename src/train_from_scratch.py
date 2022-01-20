import pickle
import time

import numpy as np
from tqdm import tqdm

from src.models.ModelFromScratch import ModelFromScratch
from src.DataManager import DataManager
from config import FULL_UTK_CROPPED_PATH, FULL_UTK_CROPPED_METADATA_FILENAME

INPUT_SIZE = (124, 124, 3)

N_SAMPLE = 1
MODE = 'evaluate'

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
if MODE != 'evaluate':
    print('Read training images ...')
    X_train, y_train = data_manager.get_X(train), data_manager.get_y(train)
    print('Read validation images ...')
    X_val, y_val = data_manager.get_X(validation), data_manager.get_y(validation)
print('Read test images ...')
X_test, y_test = data_manager.get_X(test), data_manager.get_y(test)

model_from_scratch = ModelFromScratch(INPUT_SIZE)

if MODE == 'tran':
    model_from_scratch.train(X_train, y_train, X_val, y_val, X_test, y_test)
elif MODE == 'search':
    model_from_scratch.hyperparameter_optimization(X_train, y_train, X_val, y_val, X_test, y_test, max_trials=10)
elif MODE == 'evaluate':
    # Load model
    model_from_scratch.load_weights()
    # Evaluate performances
    res = model_from_scratch.evaluate(X_test, y_test, dump=False)
    loss, gender_output_loss, age_output_loss, gender_output_accuracy, age_output_mean_absolute_error = res
    res_dict = dict(loss=loss,
                    gender_output_loss=gender_output_loss,
                    age_output_loss=age_output_loss,
                    gender_output_accuracy=gender_output_accuracy,
                    age_output_mean_absolute_error=age_output_mean_absolute_error)
    # Evaluate predictions times
    times = []

    for i in tqdm(X_test):
        img = np.expand_dims(i, 0)
        start = time.time()
        _ = model_from_scratch.predict(img)
        end = time.time()
        times.append(end - start)
    res_dict['times'] = times
    # Dump results
    with open('../doc/reports/model_from_scratch_evaluation.pickle', 'wb') as f:
        pickle.dump(res_dict, f)
