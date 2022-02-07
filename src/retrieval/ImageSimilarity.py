import os
import pickle
from os.path import join

import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from src.models.Model import Model as MyModel
from src.DataManager import DataManager
import numpy as np

import csv


class ImageSimilarity:

    CELEBRITIES_IMAGES_PATH = join('..', 'dataset', 'Retrieval', 'images')
    CELEBRITIES_METADATA_PATH = join('..', 'dataset', 'Retrieval', 'wiki_final.pickle')
    FEATURES_PATH = join('..', 'dataset', 'CELEBS', 'features.pickle')

    def __init__(self,
                 images_path=CELEBRITIES_IMAGES_PATH,
                 features_path=FEATURES_PATH,
                 metadata_path=CELEBRITIES_METADATA_PATH):
        self.images_path = images_path
        self.metadata: pd.DataFrame = self.read_metadata(metadata_path)
        self.features_path = features_path
        self.features = None

    def read_metadata(self, metadata_path) -> pd.DataFrame:
        df = pd.read_pickle(metadata_path)
        df = df.set_index('id')
        return df

    def load_features(self):
        with open(self.features_path, 'rb') as handle:
            self.features = pickle.load(handle)

    def generate_features(self, model: MyModel):
        features = {}
        for celeb_id, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            img_path = join(self.images_path, row['filename'])
            input_shape = model.get_input_shape()
            img = DataManager.read_image(img_path, input_shape, normalize=True, crop=False)
            img = np.expand_dims(img, 0)
            # Extract features
            features[celeb_id] = model.extract_features(img)

        # Dump features
        with open(self.features_path, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    def find_most_similar(self, my_feature, metric_distance=cosine_similarity):

        distances = {}

        for i, other_feature in self.features.items():
            #d = metric_distance(np.expand_dims(my_feature, 0), np.expand_dims(other_feature, 0))
            d = metric_distance(my_feature,other_feature)


            distances[i] = d.flatten()[0]

        min_id, _ = min(distances.items(), key=lambda x: x[1])
        return min_id, self.metadata.loc[min_id]
