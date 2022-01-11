import os

from PIL import ImageFont

from src.detection.yolo.YoloFaceDetector import YoloFaceDetector
from PIL import Image

from os.path import join

yolo_face_detector = YoloFaceDetector()

p = '/Users/gianlucagiudice/Documents/uni/visual-information-processing/visual-processing-project/dataset/FDDB/originalPics/2002/07/19/big/'
f = [join(p, x) for x in os.listdir(p)]

for img in f:
    font = ImageFont.load_default()
    image = Image.open(img)
    position, confidence = yolo_face_detector.detect_image(image)
    #TODO: Prendo quella con più confidenza