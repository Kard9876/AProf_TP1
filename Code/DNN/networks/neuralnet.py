import math
import matplotlib

import numpy as np
from matplotlib import pyplot as plt

from Code.DNN.functions.mse import MeanSquaredError
from Code.DNN.functions.metrics import mse

matplotlib.use('TkAgg')

class NeuralNetwork:
    def __init__(self, epochs=100, batch_size=128, optimizer=None, regulator=None, verbose=False, loss=MeanSquaredError(),
                 metric: callable = mse, patience=-1, min_delta=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self._regulator = regulator

        # attributes
        self.layers = []
        self.train_history = {}
        self.validation_history = {}

        self._early_stop = patience > 0
        self._patience = patience
        self._patience_counter = 0
        self._min_delta = min_delta
        self._best_loss = math.inf

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X

        for layer in self.layers:
            output = layer.forward_propagation(output, training)

        return output

    def backward_propagation(self, output_error):
        error = output_error

        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, self._regulator)

        return error

    def fit(self, X, y, X_val=None, y_val=None):

        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.train_history = {}
        self.validation_history = {}

        break_val = False

        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []

            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)

                # Backward propagation
                error = self.loss.derivative(y_batch, output)

                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.function(y_all, output_x_all)

            metric_s = 'NA'
            metric = 'NA'

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"

            # save loss and metric for each epoch
            self.train_history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

            # Early Stop. Implementation based of: https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf
            if self._early_stop:
                if loss < self._best_loss - self._min_delta:
                    self._best_loss = loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

                    if self._patience_counter >= self._patience:
                        print(f"Early stopping at epoch {epoch}")
                        break_val = True

            if X_val is not None and y_val is not None:
                # store mini-batch data for epoch loss and quality metrics calculation
                val_output_x_ = []
                val_y_ = []

                for val_X_batch, val_y_batch in self.get_mini_batches(X_val, y_val):
                    # Forward propagation
                    val_output = self.forward_propagation(val_X_batch, training=True)

                    """
                    # Backward propagation
                    val_error = self.loss.derivative(val_y_batch, val_output)

                    self.backward_propagation(val_error)
                    """

                    val_output_x_.append(val_output)
                    val_y_.append(val_y_batch)

                val_output_x_all = np.concatenate(val_output_x_)
                val_y_all = np.concatenate(val_y_)

                # compute loss
                val_loss = self.loss.function(val_y_all, val_output_x_all)

                val_metric = 'NA'

                if self.metric is not None:
                    val_metric = self.metric(val_y_all, val_output_x_all)

                # save loss and metric for each epoch
                self.validation_history[epoch] = {'loss': val_loss, 'metric': val_metric}

            if break_val:
                break

        return self

    def predict(self, X):
        return self.forward_propagation(X, training=False)

    def score(self, y, predictions):
        if self.metric is None:
            raise ValueError("No metric specified for the neural network.")

        return self.metric(y, predictions)

    def plot_train_curves(self):
        epochs = self.epochs + 1

        training_accuracy = [0] * epochs
        validation_accuracy = [0] * epochs

        training_loss = [0] * epochs
        validation_loss = [0] * epochs

        for i in range(1, self.epochs + 1):
            training_accuracy[i] = self.train_history[i]['metric']
            training_loss[i] = self.train_history[i]['loss']

            validation_accuracy[i] = self.validation_history[i]['metric']
            validation_loss[i] = self.validation_history[i]['loss']

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
