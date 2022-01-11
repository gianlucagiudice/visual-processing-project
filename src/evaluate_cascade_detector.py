import cv2
from src.detection.cascade.CascadeFaceDetector import CascadeFaceDetector

import os
from os.path import join


cascade_face_detector = CascadeFaceDetector()



path = 'C:/Users/Luca/Desktop/Universita/Magistrale/Lab_Visual/Progetto/FDDB/originalPics/2002/07/19/big'
files = [x for x in os.listdir(path)]

for f in files:
    p = join(path, f)

    img = cv2.imread(p)

    faces = cascade_face_detector.detect_image(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow(f, img)

    try:
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        break