import os
import pickle
from os.path import join

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from src.models.Model import Model as MyModel
from src.DataManager import DataManager
import numpy as np

import csv


class ImageSimilarity:

    CELEBRITIES_IMAGES_PATH = join('..', 'dataset', 'CELEBS', 'images')
    CELEBRITIES_NAMES_PATH = join('..', 'dataset', 'CELEBS', 'celebrities.csv')
    FEATTURES_PATH = join('..', 'dataset', 'CELEBS', 'features.pickle')

    def __init__(self,
                 images_path=CELEBRITIES_IMAGES_PATH,
                 features_path=FEATTURES_PATH,
                 names_path=CELEBRITIES_NAMES_PATH):
        self.images_path = images_path
        self.features_path = features_path
        self.names_path = names_path
        self.names_dict = self.read_names_dict(self.names_path)
        self.features = None

    def read_names_dict(self, names_path):
        with open(names_path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            name_dict = {int(id_celeb): name_celeb for id_celeb, name_celeb in spamreader}
        return name_dict

    def load_features(self):
        with open(self.features_path, 'rb') as handle:
            self.features = pickle.load(handle)

    def generate_features(self, model: MyModel):
        features = {}
        for image in tqdm(os.listdir(self.images_path)):
            img_path = join(self.images_path, image)
            input_shape = model.get_input_shape()
            img = DataManager.read_image(img_path, input_shape, normalize=True)
            img = np.expand_dims(img, 0)
            # Extract feature
            celeb_id, _ = image.split('.')
            features[int(celeb_id)] = model.extract_feature(img)

        # Dump features
        with open(self.features_path, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    def find_most_similar(self, image, model, metric_distance=cosine_similarity):
        my_feature = model.extract_feature(np.expand_dims(image, 0))

        distances = {}
        for i, other_feature in self.features.items():
            d = metric_distance(np.expand_dims(my_feature, 0), np.expand_dims(other_feature, 0))
            distances[i] = d.flatten()[0]

        min_id, _ = min(distances.items(), key=lambda x: x[1])
        return min_id, self.names_dict[min_id]
