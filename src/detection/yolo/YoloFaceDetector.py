from libs.yolo3_keras.yolo import YOLO
from os.path import join


class YoloFaceDetector:

    DEFAULT_MODEL_PATH = join('..', 'model', 'yolo_finetuned_best.h5')
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
        res_swapped = []

        for box in res[0]:
            a, b, c, d = box
            res_swapped.append([b, a, d, c])

        res = list(zip(res_swapped, res[1]))
        res_sorted = sorted(list(res), key=lambda x: x[1], reverse=True)
        return res_sorted