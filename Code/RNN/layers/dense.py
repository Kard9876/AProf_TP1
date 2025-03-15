import numpy as np
import copy

from .layer import Layer


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        super().__init__()
        self.b_opt = None
        self.w_opt = None

        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        limit = np.sqrt(6 / (self._input_shape[1] + self.n_units))
        # limit = 1 / np.sqrt(self._input_shape[1])
        self.weights = np.random.uniform(-limit, limit, (self.input_shape()[1], self.n_units))

        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs

        batch_size, timesteps, input_dim = inputs.shape  # Assuming shape (batch_size, timesteps, features)
        output_sequence = np.zeros((batch_size, timesteps, self.n_units))
        for t in range(timesteps):
            timestep_input = inputs[:, t, :]
            timestep_output = np.dot(timestep_input, self.weights) + self.biases
            output_sequence[:, t, :] = timestep_output

        reshaped_weights = np.expand_dims(self.weights, axis=0)
        tiled_weights = np.tile(reshaped_weights, (batch_size, 1, 1))

        # Assuming that each word gives the probability of being generated, the probability of the whole phrase being generated is weighted average of all predictions
        output_sequence = np.average(output_sequence, weights=tiled_weights, axis=1)

        return output_sequence

    def backward_propagation(self, output_error, regulator=None):
        batch_size, output_dim = output_error.shape
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            # Aggregate input across timesteps
            aggregated_input = np.sum(self.input[b], axis=0)  # sum input over timesteps

            # Calculate gradients
            grad_weights += np.dot(aggregated_input.reshape(-1, 1), output_error[b, :].reshape(1, -1))
            grad_biases += output_error[b, :]
            grad_input[b] = np.dot(output_error[b, :].reshape(1, -1), self.weights.T)

        # Gradient Clipping (Optional but recommended)
        np.clip(grad_weights, -1, 1, out=grad_weights)
        np.clip(grad_biases, -1, 1, out=grad_biases)
        np.clip(grad_input, -1, 1, out=grad_input)

        # Update weights and biases (using an optimizer)
        self.weights = self.w_opt.update(self.weights, grad_weights)
        self.biases = self.b_opt.update(self.biases, grad_biases)

        return grad_input

    def output_shape(self):
        return (self.n_units,)
