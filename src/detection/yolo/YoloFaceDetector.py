from libs.yolo3_keras.yolo import YOLO
from os.path import join


class YoloFaceDetector:

    DEFAULT_MODEL_PATH = join('..', 'model', 'trained_face_yolo.h5')
    DEFAULT_CLASSES_PATH = join('..', 'libs', 'yolo3_keras', 'model_data', 'fddb_classes.txt')
    DEFAULT_ANCHORS_PATH =  join('..', 'libs', 'yolo3_keras', 'model_data', 'yolo_anchors.txt')

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 classes_path=DEFAULT_CLASSES_PATH,
                 anchors_path=DEFAULT_ANCHORS_PATH):
        args_dict = dict(image=True, model=model_path, classes=classes_path, anchors_path=anchors_path)
        self.yolo_model = YOLO(**args_dict)

    def detect_image(self, image):
        res = self.yolo_model.detect_image(image, verbose=False)
        res = list(zip(res[0], res[1]))
        res_sorted = sorted(list(res), key=lambda x: x[1], reverse=True)
        if res_sorted:
            return res_sorted[0]
        else:
            return None, None
