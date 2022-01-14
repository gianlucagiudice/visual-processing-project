import cv2
from os.path import join
import os


DEFAULT_MODEL_PATH = os.path.abspath('../detection/cascade/model/faceDetector_FDDB_LBP_10_0.01.xml')
DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH.replace('\\', '/')
DEFAULT_MODEL_PATH = 'C:/Users/Luca/Desktop/visual/src/detection/cascade/model/faceDetector_FDDB_LBP_10_0.01.xml'
print(f'Local path:', DEFAULT_MODEL_PATH)

class CascadeFaceDetector:

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH):

        self.cascade_model = cv2.CascadeClassifier(model_path)

    def detect_image(self, image):
        faces = self.cascade_model.detectMultiScale(image)
        return faces
