import cv2
import pandas as pd
from os.path import join

df = pd.read_pickle('/Users/gianlucagiudice/Downloads/imdb.pickle')

df = df.sample(frac=1).reset_index(drop=True)

for _, row in df.iterrows():
    path = join('../dataset/imdb_crop/', row.full_path)
    img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

