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
    FEATURES_PATH = join('..', 'dataset', 'Retrieval', 'features.pickle')

    def __init__(self,
                 images_path=CELEBRITIES_IMAGES_PATH,
                 features_path=FEATURES_PATH,
                 metadata_path=CELEBRITIES_METADATA_PATH):
        self.images_path = images_path
        self.metadata: pd.DataFrame = self.read_metadata(metadata_path)
        self.features_path = features_path
        self.features = None
        self.features_male = None
        self.features_female = None
        self.id_to_age = None

    def read_metadata(self, metadata_path) -> pd.DataFrame:
        df = pd.read_pickle(metadata_path)
        df = df.set_index('id')
        return df

    def load_features(self, age_thd=90):
        with open(self.features_path, 'rb') as handle:
            self.features_dict = pickle.load(handle)

        # Remove invalid rows
        invalid_rows = self.metadata[self.metadata['age'] > age_thd].index
        self.metadata = self.metadata.drop(invalid_rows)
        self.features_dict = dict([(key, value) for key, value in self.features_dict.items() if key not in invalid_rows])

        # Get max age
        self.max_age = self.metadata['age'].max()

        # Flip gender metadata
        remap_gender = {1: 0, 0: 1}
        self.metadata['gender'] = self.metadata['gender'].map(lambda x: remap_gender[x])

        # Extract Male/Gender features
        self.features, self.features_id, self.features_age, self.features_matrix = \
            self.extract_features(self.metadata.index)

        # Extract male
        male_ids = self.metadata.query('gender==0').index
        self.features_male, self.features_male_id, self.features_male_age, self.features_male_matrix = \
            self.extract_features(male_ids)

        # Extract female
        female_ids = self.metadata.query('gender==1').index
        self.features_female, self.features_female_id, self.features_female_age, self.features_female_matrix = \
            self.extract_features(female_ids)

        # Extract id to age dict
        self.id_to_age = dict(zip(self.metadata['age'].index, self.metadata['age']))

    def extract_features(self, ids):
        f_dict = dict([(key, value) for key, value in self.features_dict.items() if key in ids])
        features_id, features = zip(*f_dict.items())
        features_age = self.metadata.loc[ids]['age'].values
        features_matrix = np.concatenate([v / np.sqrt(np.sum(v**2)) for v in features]).T

        return features, features_id, features_age, features_matrix

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

    def find_most_similar(
            self,
            my_feature,
            gender=None,
            age=None,
            weight_features=3,
            weight_age=1,
            metric_distance=cosine_similarity,
            optimize=True
    ):

        if gender == 0:
            features, features_id, ages, features_matrix = \
                self.features_male, self.features_male_id, self.features_male_age, self.features_male_matrix
        elif gender == 1:
            features, features_id, ages, features_matrix = \
                self.features_female, self.features_female_id, self.features_female_age, self.features_female_matrix
        else:
            features, features_id, ages, features_matrix = \
                self.features, self.features_id, self.features_age, self.features_matrix

        args = dict(
            age=age,
            weight_features=weight_features,
            weight_age=weight_age,
            metric_distance=metric_distance
        )

        if not optimize:
            actor_id, actor_metadata, score = self.naive_find_most_similar(
                my_feature, features, features_id, ages, **args)
        else:
            args.pop('metric_distance')
            actor_id, actor_metadata, score = self.optimized_find_most_similar(
                my_feature, features_matrix, features_id, ages, **args)

        return actor_id, actor_metadata, score

    def naive_find_most_similar(
            self,
            my_feature,
            features,
            features_id,
            ages,
            age=None,
            weight_features=1,
            weight_age=0,
            metric_distance=cosine_similarity):

        similarities = np.zeros(len(features))
        for i, other_feature in enumerate(features):
            similarities[i] = metric_distance(my_feature, other_feature).flatten()

        if age:
            age_distances = np.absolute(age - ages) / self.max_age * weight_age
            final_similarities = weight_features * similarities - weight_features * age_distances
        else:
            final_similarities = similarities

        max_pos = np.argmax(final_similarities)
        return features_id[max_pos], self.metadata.loc[features_id[max_pos]], similarities[max_pos]

    def optimized_find_most_similar(
            self,
            my_feature,
            features_matrix,
            features_id,
            ages,
            age=None,
            weight_features=1,
            weight_age=0):

        my_feature = my_feature / np.sqrt(np.sum(my_feature**2))
        similarities = np.dot(my_feature, features_matrix).flatten()

        if age:
            age_distances = np.absolute(age - ages) / self.max_age * weight_age
            final_similarities = weight_features * similarities - weight_features * age_distances
        else:
            final_similarities = similarities

        max_pos = np.argmax(final_similarities)

        return features_id[max_pos], self.metadata.loc[features_id[max_pos]], similarities[max_pos]
