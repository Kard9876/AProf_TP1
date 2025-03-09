import numpy as np
from scipy import optimize

from Code.LogisticRegression.utils.sigmoid import sigmoid


class LogisticRegression:
    def __init__(self, n_features):
        self._n_features = n_features
        # self.theta = np.zeros(n_features)
        limit = 1 / np.sqrt(n_features)
        self.theta = np.random.uniform(-limit, limit, n_features)

    def probability(self, instance):
        x = np.empty([self._n_features])

        x[0] = 1
        x[1:] = np.array(instance[:self._n_features - 1])

        return sigmoid(np.dot(self.theta, x))

    def predict(self, instance):
        p = self.probability(instance)

        res = 0

        if p >= 0.5:
            res = 1

        return res

    def predict_many(self, X):
        p = sigmoid(np.dot(X, self.theta))
        return np.where(p >= 0.5, 1, 0)

    def cost_function(self, X, y, theta=None):
        if theta is None:
            theta = self.theta

        m = X.shape[0]
        p = sigmoid(np.dot(X, theta))

        cost = (-y * np.log(p) - (1 - y) * np.log(1 - p))
        res = np.sum(cost) / m

        return res

    def gradient_descent(self, X, y, alpha=0.01, iters=10000):
        m = X.shape[0]
        n = X.shape[1]

        self.theta = np.zeros(n)

        for its in range(iters):
            J = self.cost_function(X, y)

            if its % 10 == 0:
                print(J)

            delta = X.T.dot(sigmoid(X.dot(self.theta)) - y)
            self.theta -= (alpha / m * delta)

    def build_model(self, X, y, maxiter, maxfun):
        self.optim_model(X, y, maxiter, maxfun)

    def optim_model(self, X, y, maxiter, maxfun):
        n = X.shape[1]
        options = {'maxiter': maxiter, 'maxfun': maxfun, 'full_output': True, 'disp': True, 'ftol': 1}

        initial_theta = np.zeros(n)

        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.cost_function(X, y, theta), initial_theta,
                                               **options)

    def print_coefs(self):
        print(self.theta)
