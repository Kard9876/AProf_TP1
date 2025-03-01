import numpy as np
from scipy import optimize

from Code.LogisticRegression.utils.sigmoid import sigmoid


class LogisticRegression:
    def __init__(self, n_features):
        self._n_features = n_features
        self.theta = self.theta = np.zeros(n_features)

    def probability(self, instance):
        x = np.empty([self._n_features])

        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1] - 1])

        return sigmoid(np.dot(self.theta, x))

    def predict(self, instance):
        p = self.probability(instance)

        res = 0

        if p >= 0.5:
            res = 1

        return res

    def costFunction(self, X, y, theta=None):
        if theta is None: theta = self.theta

        m = X.shape[0]
        p = sigmoid(np.dot(X, theta))

        cost = (-y * np.log(p) - (1 - y) * np.log(1 - p))
        res = np.sum(cost) / m

        return res

    def gradientDescent(self, X, y, alpha=0.01, iters=10000):
        m = X.shape[0]
        n = X.shape[1]

        self.theta = np.zeros(n)

        for its in range(iters):
            J = self.costFunction()

            if its % 1000 == 0:
                print(J)

            delta = X.T.dot(sigmoid(X.dot(self.theta)) - y)
            self.theta -= (alpha / m * delta)

    def buildModel(self):
        self.optim_model()

    def optim_model(self, X, y):
        n = X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta, X, y), initial_theta, **options)

    def printCoefs(self):
        print(self.theta)
