import math
from datetime import datetime
from os.path import join

import cv2
import numpy as np
import pandas as pd
from skimage import feature, color
from skimage.feature import hog
from tqdm import tqdm

from src.EnhancementUtils import EnhancementUtils
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
        self.enhancement = EnhancementUtils()

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

    def extract_dataset_features(self, X, y, compute_sift=True, compute_hog=True, compute_hist=True, compute_lbp=True):
        df = pd.DataFrame()

        print('Extracting dataset features ...')
        with tqdm(total=len(X)) as pbar:
            for x in X:
                x = self.enhancement.equalize_histogram(x)
                x = self.enhancement.bilateral_filter(x)
                features = self.extract_features(x, compute_sift, compute_hog, compute_hist, compute_lbp)
                df = df.append(features, ignore_index=True)
                pbar.update(1)

        df["gender"] = y["gender"].values
        df["age"] = y["age"].values

        return df

    def extract_features(self, img, compute_sift=True, compute_hog=True, compute_hist=True, compute_lbp=True):
        # to grey
        grey = color.rgb2gray(img)
        # division in 4 parts of original img
        img_parts = self.crop_image_4(img)
        # division in parts of grey img
        grey_parts = []
        for part in img_parts:
            grey_parts.append(color.rgb2gray(part))

        # vector of features
        df = {}
        i = 0

        if compute_sift:
            # SIFT - on the entire face
            _, descriptors = self.extract_sift(grey, self.n_sift)
            img_sift = [d.tolist() for d in descriptors]
            for s in img_sift:
                for el in s:
                    df[i] = el
                    i = i + 1

        if compute_hog:
            # HOG - on the entire face
            img_hog = self.extract_hog(img)
            for h in img_hog:
                df[i] = h
                i = i + 1

        if compute_hist:
            # color histogram (4 lists of 3 histograms)
            img_color_hist = []
            for part in img_parts:
                img_color_hist.extend(self.color_histogram(part, self.color_hist_bins))
            for h in img_color_hist:
                for el in h:
                    df[i] = el
                    i = i + 1

        if compute_lbp:
            img_lbp = self.compute_lbp(grey, self.lbp_n_points, self.lbp_radius)
            for h in img_lbp:
                df[i] = h
                i = i + 1

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
        # normalize
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        cv2.normalize(hist, hist)
        hist.flatten()

        return hist

    @staticmethod
    def color_histogram(img, bins=256):
        histograms = []
        for ch in range(3):
            histogram, _ = np.histogram(img[:, :, ch], bins=bins, range=(0, 256))
            histograms.append(histogram.tolist())

        # normalize
        for i in range(len(histograms)):
            histograms[i] = [x / sum(histograms[i]) for x in histograms[i]]

        return histograms

    @staticmethod
    def extract_sift(img, n_sift):
        detector = cv2.SIFT_create(nfeatures=n_sift)
        # detect features from the image
        image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = detector.detectAndCompute(image8bit, None)
        # normalize descriptors
        for i in range(len(descriptors)):
            descriptors[i] = [x / sum(descriptors[i]) for x in descriptors[i]]

        return keypoints, descriptors

    @staticmethod
    def extract_hog(img):
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)

        return fd
