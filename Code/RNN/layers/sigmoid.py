import numpy as np

from .activation import ActivationLayer


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        f_x = self.activation_function(input)
        return f_x * (1 - f_x)
