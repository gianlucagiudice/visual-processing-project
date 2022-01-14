import math

import cv2
import numpy as np


class EnhancementUtils:

    def __init__(self):
        pass

    def equalize_histogram(self, rgb_img):
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        return equalized_img

    def bilateral_filter(self, img, d=15, sigmaColor=25, sigmaSpace=25):
        bilateral = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return bilateral

    def is_image_too_dark(self, img, thresh=0.5):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)

        return int(np.average(v.flatten())) / 255 < thresh

    def automatic_gamma(self, img):
        # convert img to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # compute gamma = log(mid*255)/log(mean) --> it's a simple proportion!
        mid = 0.5
        mean = np.mean(gray)
        gamma = math.log(mid * 255) / math.log(mean)
        # do gamma correction
        return np.power(img, gamma).clip(0, 255).astype(np.uint8)

    def adaptive_gamma(self, img):
        y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        y = (255 * (y / 255) ** (2 ** ((128 - 128) / 128))).astype('uint8')
        img_restored = cv2.merge([y, cr, cb])
        img_restored = cv2.cvtColor(img_restored, cv2.COLOR_YCrCb2BGR)
        return img_restored
