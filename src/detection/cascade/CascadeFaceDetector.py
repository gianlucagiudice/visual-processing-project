import cv2
from os.path import join
import os


DEFAULT_MODEL_PATH = os.path.abspath('../src/detection/cascade/model/faceDetector_FDDB_LBP_10_0.01.xml')
DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH.replace('\\', '/')

print(f'Local path:', DEFAULT_MODEL_PATH)

class CascadeFaceDetector:

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH):

        self.cascade_model = cv2.CascadeClassifier(model_path)

    def detect_image(self, image):
        faces = self.cascade_model.detectMultiScale(image)

        for bbox in faces:
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]

        return faces
