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

    def bilateral_filter(self, img, d=3, sigmaColor=25, sigmaSpace=25):
        bilateral = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return bilateral

    def adaptive_gamma(self, img):
        y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        y_inv = 255 - y
        bilateral = cv2.bilateralFilter(y_inv, d=2, sigmaColor=25, sigmaSpace=25)
        enhanced = np.empty((len(y), len(y[0])))
        for i in range(len(y)):
            for j in range(len(y[0])):
                enhanced[i][j] = (255 * (y[i][j] / 255) ** (2 ** (0.5 * (128 - bilateral[i][j]) / 128)))
        enhanced = enhanced.astype('uint8')
        img_enhanced = cv2.merge([enhanced, cr, cb])

        return cv2.cvtColor(img_enhanced, cv2.COLOR_YCrCb2BGR)
