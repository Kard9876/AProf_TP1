import numpy as np
from scipy import optimize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from Code.LogisticRegression.utils.sigmoid import sigmoid


class LogisticRegression:
    def __init__(self, n_features, reg_type=None, reg_lambda=0.0):
        self._n_features = n_features
        # self.theta = np.zeros(n_features)
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        limit = 1 / np.sqrt(n_features)
        self.theta = np.random.uniform(-limit, limit, n_features)

        self._train_history = {}
        self._val_history = {}

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
        cost = (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()  # Equivalent to sum / m

        if self.reg_type == 'l2':
            reg_term = (self.reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
            cost += reg_term
        elif self.reg_type == 'l1':
            reg_term = (self.reg_lambda / m) * np.sum(np.abs(theta[1:]))
            cost += reg_term

        return cost

    def gradient_descent(self, X, y, X_val, y_val, alpha=0.01, iters=10000):
        m = X.shape[0]
        n = X.shape[1]

        self.theta = np.zeros(n)

        self._train_history = {}
        self._val_history = {}
        self._iters = iters
        for its in range(iters):
            delta = X.T.dot(sigmoid(X.dot(self.theta)) - y)
            self.theta -= (alpha / m * delta)

            J = self.cost_function(X, y)
            val_loss = self.cost_function(X, y)

            train_acc = accuracy_score(self.predict_many(X), y)
            val_acc = accuracy_score(self.predict_many(X_val), y_val)

            print(val_loss, val_acc)

            self._train_history[its] = {'loss': J, 'accuracy': train_acc}
            self._val_history[its] = {'loss': val_loss, 'accuracy': val_acc}

            if its % 10 == 0:
                print(J)

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

    def save_model(self, filename):
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename, n_features=self._n_features, theta=self.theta)

    def plot_train_curves(self):
        epochs = self._iters

        training_accuracy = [0] * epochs
        validation_accuracy = [0] * epochs

        training_loss = [0] * epochs
        validation_loss = [0] * epochs

        for i in range(0, self._iters):
            training_accuracy[i] = self._train_history[i]['accuracy']
            training_loss[i] = self._train_history[i]['loss']

            validation_accuracy[i] = self._val_history[i]['accuracy']
            validation_loss[i] = self._val_history[i]['loss'] * 100

        epochs_range = np.arange(epochs)

        plt.figure()
        plt.plot(epochs_range, training_accuracy, 'r', label='Training', )
        plt.plot(epochs_range, validation_accuracy, 'b', label='Validation')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.title('Accuracy curves')
        plt.show()

        plt.figure()
        plt.plot(epochs_range, training_loss, 'r', label='Training', )
        plt.plot(epochs_range, validation_loss, 'b', label='Validation')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.title('Loss curves')
        plt.show()
