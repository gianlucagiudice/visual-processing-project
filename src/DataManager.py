import numpy as np

import cv2
from tqdm import tqdm

from src.parameters import SEED
from sklearn.model_selection import train_test_split

from os.path import join
import pandas as pd
import math


def read_dataset_metadata(dataset_path: str, metadata_filename: str):
    df = pd.read_pickle(join(dataset_path, metadata_filename))
    df['path'] = df['full_path'].apply(lambda x: join(dataset_path, x))
    return df


def shuffle_dataset(dataset):
    return dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)


def sample_n(dataset, n_subset):
    if n_subset < 1:
        n_sample = len(dataset) * n_subset
    elif n_subset > 1:
        n_sample = n_subset
    else:
        n_sample = len(dataset)
    # Return sampled sampled
    return dataset.head(math.floor(n_sample))


class DataManager:
    X = ['path']
    y = ['gender', 'age']

    def __init__(self, dataset_path, metadata_filename, resize_shape,
                 n_subset=None, shuffle=True, test_size=0.3, validation_size=.15):
        self.dataset_path = dataset_path
        self.metadata_filename = metadata_filename
        # Train, test, validation
        self.test_size, self.validation = test_size, validation_size
        # Resize shape
        self.resize_shape = resize_shape
        # Dataset
        self.dataset = read_dataset_metadata(dataset_path, metadata_filename)
        # Shuffle dataset
        if shuffle:
            self.dataset = shuffle_dataset(self.dataset)
        # Subset dataset
        if n_subset:
            self.dataset = sample_n(self.dataset, n_subset)

    def get_dataset(self):
        return self.dataset

    def split_dataset(self, df):
        train, test = train_test_split(df, test_size=self.test_size)
        train, validation = train_test_split(train, test_size=self.validation)
        return train, validation, test

    def read_images(self, files):
        shape = (files.size, *self.resize_shape)
        images = np.empty(shape)
        # Start reading of the images
        with tqdm(total=files.size) as pbar:
            for i, image in enumerate(files):
                # Read image
                im = cv2.imread(image)
                # Change color space
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # Resize image
                im = cv2.resize(im, (self.resize_shape[0], self.resize_shape[1]))
                # Append image
                images[i] = im / 255
                # Update progress bar
                pbar.update(1)

        return images

    def get_X(self, df, return_images=True):
        files = df[DataManager.X].values.flatten()
        if not return_images:
            return files
        else:
            return self.read_images(files)

    @staticmethod
    def get_y(df):
        return df[DataManager.y]