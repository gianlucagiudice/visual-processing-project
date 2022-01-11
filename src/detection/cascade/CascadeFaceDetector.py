import cv2
from os.path import join


DEFAULT_MODEL_PATH = 'detection/cascade/model/faceDetector_FDDB_LBP_10_0.01.xml'
print(DEFAULT_MODEL_PATH)


class CascadeFaceDetector:

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH):

        self.cascade_model = cv2.CascadeClassifier(model_path)

    def detect_image(self, image):
        faces = self.cascade_model.detectMultiScale(image)
        return faces
