import numpy as np
import copy

from .layer import Layer


class DropOutLayer(Layer):

    def __init__(self, n_units, drop_rate, input_shape=None):
        super().__init__()
        self.b_opt = None
        self.w_opt = None

        self.n_units = n_units
        self._drop_rate = drop_rate
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self._tmp_weights = None

    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5

        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    # Implementation based of https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
    def _dropout(self, training):
        if not training:
            return self.weights

        m = self.weights.shape[0]
        n = self.weights.shape[1]

        if self._drop_rate == 1:
            return np.zeros(m, n)

        mask = np.random.rand(m, n) > self._drop_rate
        return mask * self.weights / (1.0 - self._drop_rate)

    def forward_propagation(self, inputs, training):
        self.input = inputs

        self._tmp_weights = self._dropout(training)
        self.output = np.dot(self.input, self._tmp_weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, regulator=None):
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self._tmp_weights.T)  # dE / dY = output error

        if regulator is not None:
            input_error += regulator.update(self.input.shape[0], self._tmp_weights)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)

        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self):
        return (self.n_units,)
