import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        """
        Initializes the Stochastic Gradient Descent (SGD) optimizer.

        Parameters:
        ----------
        learning_rate: float, optional
            The learning rate for the optimizer. Default is 0.01.
        """
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        """
        Updates the weights using the calculated gradients.

        Parameters:
        ----------
        weights: numpy.ndarray
            The current weights.
        gradients: numpy.ndarray
            The gradients of the loss function with respect to the weights.

        Returns:
        --------
        numpy.ndarray
            The updated weights.
        """
        return weights - self.learning_rate * gradients
