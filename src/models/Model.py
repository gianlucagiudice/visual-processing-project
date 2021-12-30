import numpy as np
from abc import ABC, abstractmethod


IMAGE_INPUT_SIZE = (224, 224, 3)


class Model(ABC):
    def __init__(self, input_size: np.array=IMAGE_INPUT_SIZE):
        # Model weights
        self.model = None

    @abstractmethod
    def predict(self, image: np.array) -> (bool, int):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:
        pass

    @abstractmethod
    def save_weights(self) -> None:
        # Save the weights of the model
        pass

    @abstractmethod
    def load_weights(self):
        # Load the weights of the model
        pass
