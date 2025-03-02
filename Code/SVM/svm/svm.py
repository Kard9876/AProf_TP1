# Sources:
# https://www.youtube.com/watch?v=ofB9jj6sGro
# https://github.com/SSaishruthi/SVM-using-Numpy/blob/master/SVM.ipynb
# https://medium.com/@saishruthi.tn/support-vector-machine-using-numpy-846f83f4183d

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SVM:

    def __init__(self, n_features, learning_rate, epochs, lambda_val, threshold=0.5):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self._w = np.random.rand(n_features) - 0.5
        self._l_rate = learning_rate
        self._epochs = epochs
        self._lambda = lambda_val
        self.bias = 0
        self._threshold = threshold

    def fit(self, X, y):
        self.bias = 0

        for e in range(self._epochs):
            print(f"Epoch {e+1}/{self._epochs}")

            for i, x in enumerate(X):
                val1 = np.dot(x, self._w)

                """
                if y[i] * val1 < 1:
                    self._w = self._w + self._l_rate * ((y[i] * x) - (2 * self._lambda * self._w))
                else:
                    self._w = self._w + self._l_rate * (-2 * self._lambda * self._w)
                """

                if y[i] * val1 < 1:
                    self._w -= self._l_rate * (2 * self._lambda * self._w - np.dot(x, y[i]))
                    self.bias -= self._l_rate * y[i]
                else:
                    self._w -= self._l_rate * (2 * self._lambda * self._w)

    def predict(self, X):
        # tmp = sigmoid(np.dot(X, self._w.T))
        tmp = np.dot(X, self._w.T) + self.bias
        return np.where(tmp >= self._threshold, 1, 0)
