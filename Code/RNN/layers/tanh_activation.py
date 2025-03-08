import numpy as np

from .activation import ActivationLayer


class TanhActivation(ActivationLayer):
    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        tanh_x = np.tanh(input)
        return 1 - tanh_x ** 2
