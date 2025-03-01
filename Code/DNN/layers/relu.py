import numpy as np

from .activation import ActivationLayer


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input >= 0, 1, 0)
