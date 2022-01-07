import math
from datetime import datetime
from os.path import join

import cv2
import numpy as np
from skimage import feature, color
import pandas as pd
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, LOG_DIR
from src.models.Model import IMAGE_INPUT_SIZE
from src.models.Model import Model as MyModel


class HandcraftedModel(MyModel):
    checkpoint_dir = join(CHECKPOINT_DIR, 'handcrafted')
    checkpoint_filepath = join(checkpoint_dir, 'ckpt-{epoch:03d}.hdf5')
    log_dir = join(LOG_DIR, 'fit', 'handcrafted/') + datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self, n_sift, color_hist_bins, lbp_n_points, lbp_radius, input_size=IMAGE_INPUT_SIZE):
        super().__init__(input_size)
        self.n_sift = n_sift
        self.color_hist_bins = color_hist_bins
        self.lbp_n_points = lbp_n_points
        self.lbp_radius = lbp_radius

    def predict(self, image: np.array) -> (bool, int):
        pass

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:

        # features
        self.load_features()

        # train

        # evaluate the model

        pass

    def save_weights(self) -> None:
        # Save the weights of the model
        pass

    def load_weights(self):
        # Load the weights of the model
        pass

    def load_features(self):
        pass

    def extract_dataset_features(self, X, y):
        df = pd.DataFrame()

        print('Extracting dataset features ...')
        with tqdm(total=X.shape[0]) as pbar:
            for x in X:
                features = self.extract_features(x)
                df = df.append(features, ignore_index=True)
                pbar.update(1)

        df["gender"] = y["gender"]
        df["age"] = y["age"]

        return df

    def extract_features(self, img):

        grey = color.rgb2gray(img)

        # SIFT - on the entire face
        _, descriptors = self.extract_sift(grey, self.n_sift)
        img_sift = [d.tolist() for d in descriptors]

        # division in 4 parts
        img_parts = self.crop_image_4(img)

        # color histogram (4 lists of 3 histograms)
        img_color_hist = []
        for part in img_parts:
            img_color_hist.extend(self.color_histogram(part, self.color_hist_bins))

        grey_parts = []
        for part in img_parts:
            grey_parts.append(color.rgb2gray(part))

        # LBP on grey channel - histogram
        img_lbp = []
        for grey_part in grey_parts:
            img_lbp.append(self.compute_lbp(grey_part, self.lbp_n_points, self.lbp_radius).tolist())

        # vector of features
        df = {}
        for i in range(len(img_color_hist)):  # 12 arrays of 256 numbers
            col_name = 'color_hist_' + str(i)
            df[col_name] = img_color_hist[i]
        for i in range(len(img_lbp)):  # 4 arrays of 26 numbers
            col_name = 'lbp_' + str(i)
            df[col_name] = img_lbp[i]
        for i in range(len(img_sift)):  # n arrays of 128 numbers
            col_name = 'sift_' + str(i)
            df[col_name] = img_sift[i]

        features = pd.DataFrame()
        features = features.append(df, ignore_index=True)

        return features

    @staticmethod
    def crop_image_4(img):
        w_parts = math.floor(img.shape[0] / 2)
        h_parts = math.floor(img.shape[1] / 2)

        parts = []
        if len(img.shape) == 3:
            parts.append(img[0:w_parts, 0:h_parts, :])
            parts.append(img[0:w_parts, h_parts + 1:img.shape[1], :])
            parts.append(img[w_parts + 1:img.shape[0], 0:h_parts, :])
            parts.append(img[w_parts + 1:img.shape[0], h_parts + 1:img.shape[1], :])
        else:
            parts.append(img[0:w_parts, 0:h_parts])
            parts.append(img[0:w_parts, h_parts + 1:img.shape[1]])
            parts.append(img[w_parts + 1:img.shape[0], 0:h_parts])
            parts.append(img[w_parts + 1:img.shape[0], h_parts + 1:img.shape[1]])

        return parts

    @staticmethod
    def compute_lbp(img, num_points, radius, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(img, num_points,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

    @staticmethod
    def color_histogram(img, bins=256):
        histograms = []
        for ch in range(3):
            histogram, _ = np.histogram(img[:, :, ch], bins=bins, range=(0, 256))
            histograms.append(histogram.tolist())
        return histograms

    @staticmethod
    def extract_sift(img, n_sift):
        detector = cv2.SIFT_create(nfeatures=n_sift)
        # detect features from the image
        image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = detector.detectAndCompute(image8bit, None)

        return keypoints, descriptors

