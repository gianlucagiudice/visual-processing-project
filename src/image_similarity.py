import numpy as np
import time

from src.DataManager import DataManager
from src.retrieval.ImageSimilarity import ImageSimilarity

#from src.models.ModelFromScratch import ModelFromScratch
from src.models.PretrainedVGG import PretrainedVGG

model = PretrainedVGG()
model.load_weights(feature_extractor=True)

sim = ImageSimilarity()
#sim.generate_features(model)
sim.load_features()

#img_path = '../dataset/FDDB/originalPics/2002/07/19/big/img_391.jpg'

img_path = '../dataset/Retrieval/prova/freeman.jpeg'

img = DataManager.read_image(img_path, model.get_input_shape(), normalize=True, crop=False)

img = np.expand_dims(img, 0)

features = model.predict(img)

starttime = time.time()
most_similar_id, most_similar_name, dist = sim.find_most_similar(
    features,
    gender=0,
    age=50 + 5,
    weight_features=3,
    weight_age=1,
    optimize=False
)
elapsed = time.time() - starttime
print(f'Elapsed time: {elapsed}')

starttime = time.time()
most_similar_id_2, most_similar_name_2, dist_2 = sim.find_most_similar(
    features,
    gender=0,
    age=50 + 5,
    weight_features=3,
    weight_age=1,
    optimize=True
)
elapsed = time.time() - starttime
print(f'Elapsed time: {elapsed}')

assert most_similar_id == most_similar_id_2
print(dist, dist_2)

print(f'{most_similar_id} - {most_similar_name} - {dist}')

print(f'{most_similar_name.loc["path"]}')