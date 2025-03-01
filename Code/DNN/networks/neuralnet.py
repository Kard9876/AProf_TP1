import numpy as np

from Code.DNN.functions.mse import MeanSquaredError
from Code.DNN.functions.metrics import mse


class NeuralNetwork:

    def __init__(self, epochs=100, batch_size=128, optimizer=None, verbose=False, loss=MeanSquaredError(),
                 metric: callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss = loss
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

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
            error = layer.backward_propagation(error)

        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y

        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
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
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is None:
            raise ValueError("No metric specified for the neural network.")

        return self.metric(dataset.y, predictions)
