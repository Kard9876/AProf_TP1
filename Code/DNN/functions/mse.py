import numpy as np

from .loss import Function


class MeanSquaredError(Function):

    def function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        n = y_true.size

        return (2/n) * (y_pred - y_true)
