from libs.yolo3_keras.yolo import YOLO
from os.path import join


class YoloFaceDetector:

    DEFAULT_MODEL_PATH = join('..', 'model', 'yolo_finetuned_best.h5')
    DEFAULT_CLASSES_PATH = join('..', 'libs', 'yolo3_keras', 'model_data', 'fddb_classes.txt')
    DEFAULT_ANCHORS_PATH = join('..', 'libs', 'yolo3_keras', 'model_data', 'yolo_anchors.txt')

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 classes_path=DEFAULT_CLASSES_PATH,
                 anchors_path=DEFAULT_ANCHORS_PATH):
        args_dict = dict(image=True, model=model_path, classes=classes_path, anchors_path=anchors_path)
        self.yolo_model = YOLO(**args_dict)

    def detect_image(self, image, return_confidence=True, th=0.5):
        res = self.yolo_model.detect_image(image, verbose=False)
        res_swapped = []

        for box in res[0]:
            x_min, y_min, x_max, y_max = box
            max_row, max_column = image.size
            y_min = round(max(0, y_min))
            x_min = round(max(0, x_min))
            y_max = round(min(y_max, max_column - 1))
            x_max = round(min(x_max, max_row - 1))
            res_swapped.append([y_min, x_min, y_max, x_max])

        res = list(zip(res_swapped, res[1]))
        res_sorted = sorted(list(res), key=lambda x: x[1], reverse=True)
        res_sorted = list(filter(lambda x: x[1] > th, res_sorted))

        if not return_confidence:
            res_sorted = [x[0] for x in res_sorted]

        return res_sorted
