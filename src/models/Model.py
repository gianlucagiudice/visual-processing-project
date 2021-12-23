import numpy as np
import pickle
from abc import ABC, abstractmethod


IMAGE_INPUT_SIZE = (224, 224, 3)


class Model(ABC):
    def __init__(self, input_size=IMAGE_INPUT_SIZE, train: bool=False):
        self.weights = None
        pass

    @abstractmethod
    def predict(self, image: np.array) -> (bool, int):
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def save_weights(self) -> None:
        # Save the weights of the model
        pass

    @abstractmethod
    def load_weights(self):
        # Load the weights of the model
        pass
