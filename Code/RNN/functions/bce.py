import numpy as np

from .loss import Function


class BinaryCrossEntropy(Function):
    
    def function(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return - (y_true / p) + ((1 - y_true) / (1 - p))
