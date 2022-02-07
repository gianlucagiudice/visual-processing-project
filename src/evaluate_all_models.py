import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score


from src.models.Model import Model
from src.models.ModelFromScratch import ModelFromScratch
from src.models.HandcraftedModel import HandcraftedModel
from src.models.PretrainedVGG import PretrainedVGG
from src.DataManager import DataManager
from config import FULL_UTK_CROPPED_PATH, FULL_UTK_CROPPED_METADATA_FILENAME


N_SAMPLE = 10

TEST_SIZE_FACTOR = 0.2
N_SAMPLE = N_SAMPLE / TEST_SIZE_FACTOR


def evaluate_model(input_size, target_model):
    # Read the data
    data_manager = DataManager(FULL_UTK_CROPPED_PATH,
                               FULL_UTK_CROPPED_METADATA_FILENAME,
                               input_size,
                               test_size=TEST_SIZE_FACTOR, validation_size=0.15,
                               n_subset=N_SAMPLE, normalize_images=True, normalize_age=True)
    data = data_manager.get_dataset()
    # Split into train, validation, test
    train, validation, test = data_manager.split_dataset(data)
    print('Read test images ...')
    X_test, y_test = data_manager.get_X(test), data_manager.get_y(test)
    model = target_model(input_size)
    model.load_weights()

    # Evaluate predictions times
    times = []
    predicted_gender_list = []
    predicted_age_list = []
    res_dict = dict()

    # The first prediction is always slower
    _ = model.predict(np.expand_dims(X_test[0], 0))

    for i in tqdm(X_test):
        img = np.expand_dims(i, 0)
        start = time.time()
        predicted_gender, predicted_age = model.predict(img)
        end = time.time()
        # Append predictions
        predicted_gender_list.append(predicted_gender[0])
        predicted_age_list.append(predicted_age[0])
        # Append time
        times.append(end - start)

    predicted_gender_list = np.array(predicted_gender_list)
    predicted_age_list = np.array(predicted_age_list)
    # Metrics
    inverse_true_age = data_manager.inverse_standardize_age(np.expand_dims(y_test['age'], -1))
    inverse_predicted_age = data_manager.inverse_standardize_age(predicted_age_list)
    # Save metrics
    res_dict['mae'] = mean_absolute_error(inverse_true_age, inverse_predicted_age)
    res_dict['acc'] = accuracy_score(y_test['gender'], predicted_gender_list.round())
    res_dict['top5'] = Model.compute_top_k_accuracy(5, inverse_true_age, inverse_predicted_age)
    res_dict['top10'] = Model.compute_top_k_accuracy(10, inverse_true_age, inverse_predicted_age)
    res_dict['top15'] = Model.compute_top_k_accuracy(15, inverse_true_age, inverse_predicted_age)
    res_dict['top20'] = Model.compute_top_k_accuracy(20, inverse_true_age, inverse_predicted_age)
    res_dict['times'] = times

    return res_dict


df = pd.DataFrame(columns=['model', 'mae', 'acc', 'top5', 'top10', 'top15', 'top20', 'times'])

# ---------- MODEL FROM SCRATCH ----------
print('---------- Evaluate model from scratch ----------')
res = evaluate_model(ModelFromScratch.IMAGE_INPUT_SIZE, ModelFromScratch)
res['model'] = 'from_scratch'
df.append(res, ignore_index=True)
print('\n')

# ---------- MODEL VGG  ----------
print('---------- Evaluate VGG model ----------')
res = evaluate_model(PretrainedVGG.IMAGE_INPUT_SIZE, PretrainedVGG)
res['model'] = 'vgg'
df.append(res, ignore_index=True)
print('\n')

# ---------- MODEL HANDCRAFTED ----------
print('---------- Evaluate handcrafted ----------')
res = evaluate_model(HandcraftedModel.IMAGE_INPUT_SIZE, HandcraftedModel)
res['model'] = 'handcrafted'
df.append(res, ignore_index=True)
print('\n')

