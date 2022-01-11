from src.DataManager import DataManager
from src.retrieval.ImageSimilarity import ImageSimilarity

from src.models.ModelFromScratch import ModelFromScratch

model = ModelFromScratch()
model.load_best_weights()

sim = ImageSimilarity()
#sim.generate_features(model)
sim.load_features()

img_path = '../dataset/FDDB/originalPics/2002/07/19/big/img_391.jpg'
img = DataManager.read_image(img_path, model.get_input_shape(), normalize=True)

most_similar_id, most_similar_name = sim.find_most_similar(img, model)
print(f'{most_similar_id} - {most_similar_name}')
