import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.detection.yolo.YoloFaceDetector import YoloFaceDetector
from PIL import Image

COMPUTE = True


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_score(pred, true, th = 0.5):
    d = np.zeros((len(pred), len(true)))


    for i, predicted_box in enumerate([x[0] for x in pred]):
        for j, true_box in enumerate(true):
            d[i][j] = bb_intersection_over_union(predicted_box, true_box)

    tp = sum(d.max(axis=1) > th)
    fn = len(true) - tp
    fp = len(pred) - tp

    return tp, fn, fp


def compute_metrics(df):
    tp, fn, fp, avg_time = df.sum()
    avg_time = avg_time / len(df)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return precision, recall, f1, avg_time


def evaluate_model(yolo, df):

    if COMPUTE:

        with tqdm(total=len(df)) as pbar:

            scores = []
            for i, (filename, bboxes) in df.iterrows():
                image = Image.open(os.path.join('../dataset/FDDB/', filename))

                p_time = time.time()
                pred_bboxes = yolo.detect_image(image)
                p_time = time.time() - p_time

                tp, fn, fp = compute_score(pred_bboxes, bboxes)

                scores.append([tp, fn, fp, p_time])

                pbar.update(1)

        df_scores = pd.DataFrame(scores, columns=['tp', 'fn', 'fp', 'pred_time'])
        df_scores.to_pickle('../doc/reports/scores_yolo_face.pickle')

    else:
        df_scores = pd.read_pickle('../doc/reports/scores_yolo_face.pickle')

    precision, recall, f1, avg_time = compute_metrics(df_scores)

    print(f'Precision: {precision}\n'
          f'Recall: {recall}\n'
          f'f1: {f1}\n'
          f'Avg time: {avg_time}')


if __name__ == '__main__':
    yolo_face_detector = YoloFaceDetector()
    df = pd.read_pickle('../dataset/FDDB/fddb_metadata.pickle')
    evaluate_model(yolo_face_detector, df)
