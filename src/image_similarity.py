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

img_path = '../dataset/Retrieval/Foto_Prove/Bruce_Willis.jpg'

img = DataManager.read_image(img_path, model.get_input_shape(), normalize=True, crop=False)

img = np.expand_dims(img, 0)

features = model.extract_features(img)
print(features.shape)

starttime = time.time()
most_similar_id, most_similar_name, dist = sim.find_most_similar(features)

elapsed = time.time() - starttime
print(f'{most_similar_id} - {most_similar_name} - {dist}')
print(f'Elapsed time: {elapsed}')
