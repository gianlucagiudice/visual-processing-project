import numpy as np

import cv2
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

from src.config import SEED
from sklearn.model_selection import train_test_split

from os.path import join
import pandas as pd
import math


def read_dataset_metadata(dataset_path: str, metadata_filename: str):
    df = pd.read_pickle(join(dataset_path, metadata_filename))
    df['path'] = df['full_path'].apply(lambda x: join(dataset_path, x))
    df['gender'] = df['gender'].astype(int)
    return df


def delete_nan_label_rows(dataset: pd.DataFrame, verbose=False):
    n_rows_in = len(dataset.index)
    for col in DataManager.y:
        if dataset[col].isna().sum() > 0:
            dataset_out = dataset.dropna(subset=[col])
    if verbose:
        n_rows_out = len(dataset.index)
        print('Deleted ' + str(n_rows_in - n_rows_out) + ' rows')
    return dataset_out


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


def reorder_columns(dataset, head):
    columns = dataset.columns.tolist()
    tail = set(columns) - set(head)
    ordered_columns = head + list(tail)
    # Ordered columns
    df = dataset[ordered_columns]

    return df


def remove_invalid_rows(dataset):
    len_before = len(dataset)
    print('Len before: ', len_before)
    dataset = dataset.query('age<=100')
    dataset = dataset[dataset.gender.notna()]
    dataset = dataset[dataset.age.notna()]
    len_after = len(dataset)
    print('Len after: ', len_after)
    print(f'Invalid rows: {(1 - len_after / len_before) * 100:.3f}%')

    return dataset


def are_rows_equal(rows):
    for i in range(1, len(rows)):
        if (rows[i - 1] == rows[i]).all():
            return True
    return False


def is_image_padded(img, number_equal_rows=5):
    nr = number_equal_rows
    return any([
        are_rows_equal(img[:nr, :, :]), are_rows_equal(img[-nr:, :, :]),
        are_rows_equal(img.T[:nr, :, :]), are_rows_equal(img.T[-nr:, :, :])
    ])


def is_image_too_little(img, smallest_dim):
    return img.shape[0] <= smallest_dim or img.shape[1] <= smallest_dim


def remove_invalid_images(dataset, path, smallest_dim):
    len_before = len(dataset)
    print('Len before: ', len_before)

    with tqdm(total=dataset.shape[0]) as pbar:
        for index, row in dataset.iterrows():
            img = cv2.imread(path + row.full_path)
            if is_image_too_little(img, smallest_dim=smallest_dim) or is_image_padded(img):
                dataset.drop(index, inplace=True)
            pbar.update(1)

    len_after = len(dataset)
    print('Len after: ', len_after)
    print(f'Invalid rows: {(1 - len_after / len_before) * 100:.3f}%')

    return dataset


class DataManager:
    X = ['path']
    y = ['gender', 'age']
    PADDING = .40

    def __init__(self, dataset_path, metadata_filename, resize_shape,
                 normalize_images=False, normalize_age=True,
                 n_subset=None, shuffle=True, test_size=0.3, validation_size=.15):
        self.dataset_path = dataset_path
        self.metadata_filename = metadata_filename
        # Train, test, validation
        self.test_size, self.validation = test_size, validation_size
        # Resize shape
        self.resize_shape = resize_shape
        # Dataset
        self.dataset = read_dataset_metadata(dataset_path, metadata_filename)
        # Normalize age
        if normalize_age:
            self.scaler = MinMaxScaler()
            self.dataset = self.standardize_age(self.dataset, self.scaler)
        # Shuffle dataset
        if shuffle:
            self.dataset = shuffle_dataset(self.dataset)
        # Subset dataset
        if n_subset:
            self.dataset = sample_n(self.dataset, n_subset)
        # Normalize images
        self.normalize_images = normalize_images

    def get_dataset(self):
        return self.dataset

    def split_dataset(self, df):
        train, test = train_test_split(df, test_size=self.test_size)
        train, validation = train_test_split(train, test_size=self.validation)
        return train, validation, test

    def read_images(self, files, crop=False):
        shape = (files.size, *self.resize_shape)
        images = np.empty(shape)
        # Start reading of the images
        with tqdm(total=files.size) as pbar:
            for i, image in enumerate(files):
                # Append image
                images[i] = self.read_image(image, self.resize_shape, normalize=self.normalize_images, crop=crop)
                # Update progress bar
                pbar.update(1)

        return images

    @staticmethod
    def read_image(image, resize_shape, normalize, crop):
        # Read image
        im = cv2.imread(image)
        # Change color space
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # Remove padding
        if crop:
            im = DataManager.crop_image(im)
        # Resize image
        im = cv2.resize(im, (resize_shape[0], resize_shape[1]))
        # Normalize image
        if normalize:
            im = im / 255
            im = im.astype(np.float32)
        return im

    @staticmethod
    def crop_image(im, padding=PADDING):
        height, width, _ = im.shape
        ratio = 1 / (1 + padding)

        top_y = height-math.floor(height*ratio)
        bottom_y = math.floor(height*ratio)
        right_x = math.floor(width*ratio)
        left_x = width - math.floor(width*ratio)
        return im[top_y:bottom_y, left_x:right_x, :]

    def get_X(self, df, return_images=True):
        files = df[DataManager.X].values.flatten()
        if not return_images:
            return files
        else:
            return self.read_images(files)

    @staticmethod
    def get_y(df):
        return df[DataManager.y]

    def standardize_age(self, dataset, scaler):
        x = np.expand_dims(dataset['age'], -1)
        scaler.fit(x)
        new_x = scaler.transform(x)
        dataset['age'] = new_x
        return dataset

    def inverse_standardize_age(self, ages):
        return self.scaler.inverse_transform(ages)

    def delete_nan_columns(self, df_train, df_val, df_test):
        n_col_in = len(df_train.columns)

        for df in (df_train, df_val, df_test):
            for col in df.columns:
                if df[col].isna().sum() > 0 and col != 'gender' and col != 'age':
                    df_train.drop(col, inplace=True, axis=1, errors='ignore')
                    df_val.drop(col, inplace=True, axis=1, errors='ignore')
                    df_test.drop(col, inplace=True, axis=1, errors='ignore')

        n_col_out = len(df_train.columns)
        print('Deleted a maximum of ' + str(n_col_in - n_col_out) + ' columns')
