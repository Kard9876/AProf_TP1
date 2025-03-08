import numpy as np
import copy

from .layer import Layer


class DenseLayer(Layer):

    def __init__(self, n_units, timestep, input_shape=None):
        super().__init__()
        self.b_opt = None
        self.w_opt = None

        self.n_units = n_units
        self._timestep = timestep
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        limit = 1 / np.sqrt(self._input_shape[1])
        self.weights = np.random.uniform(-limit, limit, (self.input_shape()[0] // self._timestep, self.input_shape()[1], self.n_units))

        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs

        timesteps = inputs.shape[1]  # Assuming shape (batch_size, timesteps, features)
        output_sequence = []
        for t in range(timesteps):
            timestep_input = inputs[:, t, :]
            timestep_output = np.dot(timestep_input, self.weights) + self.biases
            output_sequence.append(timestep_output)

        return np.stack(output_sequence, axis=1).squeeze(-1)

    def backward_propagation(self, output_error, regulator=None):
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)  # dE / dY = output error

        if regulator is not None:
            input_error += regulator.update(self.input.shape[0], self.weights)

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
        """

        batch_size, timesteps, output_dim = output_error.shape
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.input)

        for t in range(timesteps):
            timestep_grad = output_error[:, t, :]
            timestep_input = self.input[:, t, :]

            # Calculate gradients
            grad_weights += np.dot(timestep_input.T, timestep_grad)
            grad_biases += np.sum(timestep_grad, axis=0)
            grad_input[:, t, :] = np.dot(timestep_grad, self.weights.T.squeeze(-1))

        # Update weights and biases (using an optimizer)
        self.weights = self.w_opt.update(self.weights, grad_weights)
        self.biases = self.b_opt.update(self.biases, grad_biases)

        return grad_input

    def output_shape(self):
        return (self.n_units,)
