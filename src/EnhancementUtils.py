import cv2
import numpy as np


class EnhancementUtils:

    def __init__(self):
        pass

    def equalize_histogram(self, rgb_img):
        # takes a normalized image!
        rgb_img = np.uint8(rgb_img * 255)
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        return equalized_img / 255

    def bilateral_filter(self, img, d=15, sigmaColor=25, sigmaSpace=25):
        # takes a normalized image!
        img = np.uint8(img * 255)
        bilateral = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return bilateral / 255
