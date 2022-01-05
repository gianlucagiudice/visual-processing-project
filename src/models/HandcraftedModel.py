import math
from datetime import datetime
from os.path import join

import cv2
import numpy as np
from skimage import feature, color

from src.config import CHECKPOINT_DIR, LOG_DIR
from src.models.Model import IMAGE_INPUT_SIZE
from src.models.Model import Model as MyModel


class HandcraftedModel(MyModel):
    checkpoint_dir = join(CHECKPOINT_DIR, 'handcrafted')
    checkpoint_filepath = join(checkpoint_dir, 'ckpt-{epoch:03d}.hdf5')
    log_dir = join(LOG_DIR, 'fit', 'handcrafted/') + datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self, input_size=IMAGE_INPUT_SIZE):
        super().__init__(input_size)

        # define the model

        # save the model

    def predict(self, image: np.array) -> (bool, int):
        pass

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:

        # feature extraction
        for img in x_train:
            img_features = self.extract_features(img)
            # append to x_train_features and do the same with test and val

        # train

        # evaluate the model

        pass

    def save_weights(self) -> None:
        # Save the weights of the model
        pass

    def load_weights(self):
        # Load the weights of the model
        pass

    def extract_features(self, img):

        features = []

        # HAAR - on the entire face, preconfigured cascade detectors
        # TODO: tuning!!!
        face_rect, eye_rect = self.extract_haar(img)
        img_haar = face_rect.tolist()
        img_haar.extend(eye_rect.tolist())

        grey = color.rgb2gray(img)

        # SIFT - on the entire face
        _, descriptors = self.extract_sift(grey)
        img_sift = [d.tolist() for d in descriptors]

        # division in 4 parts
        img_parts = self.crop_image_4(img)

        # color histogram (4 lists of 3 histograms)
        img_color_hist = []
        for part in img_parts:
            img_color_hist.extend(self.color_histogram(part))

        grey_parts = []
        for part in img_parts:
            grey_parts.append(color.rgb2gray(part))

        # LBP on grey channel - histogram
        img_lbp = []
        # TODO: parameters to be tuned!!!
        num_points = 24
        radius = 8
        for grey_part in grey_parts:
            img_lbp.append(self.compute_lbp(grey_part, num_points, radius).tolist())

        # vector of features
        features.extend(img_haar)  # n bbox arrays (4 points for every bbox)
        features.extend(img_color_hist)  # 12 arrays of 256 numbers
        features.extend(img_lbp)  # 4 arrays of 26 numbers
        features.extend(img_sift)  # n arrays of 128 numbers

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
    def color_histogram(img):
        histograms = []
        for ch in range(3):
            histogram, _ = np.histogram(img[:, :, ch], bins=256, range=(0, 256))
            histograms.append(histogram.tolist())
        return histograms

    @staticmethod
    def extract_sift(img):
        detector = cv2.SIFT_create()
        # detect features from the image
        image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = detector.detectAndCompute(image8bit, None)

        return keypoints, descriptors

    @staticmethod
    def extract_haar(img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        face_rect = face_cascade.detectMultiScale(img,
                                                  scaleFactor=1.2,
                                                  minNeighbors=5)

        eye_rect = eye_cascade.detectMultiScale(img,
                                                scaleFactor=1.2,
                                                minNeighbors=5)

        return face_rect, eye_rect
