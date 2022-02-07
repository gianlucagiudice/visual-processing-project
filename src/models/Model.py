import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, input_size: np.array):
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

    @staticmethod
    def compute_top_k_accuracy(k, age, age_preds):
        top_k_acc = 0

        for i in range(len(age)):
            if abs(age[i] - age_preds[i]) <= k:
                top_k_acc += 1

        return top_k_acc / len(age)
