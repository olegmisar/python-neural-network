import numpy as np
import scipy.special
import json

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_ih = np.random.normal(0.0, pow(input_nodes, -0.5), (hidden_nodes, input_nodes))
        self.weights_ho = np.random.normal(0.0, pow(hidden_nodes, -0.5), (output_nodes, hidden_nodes))
        self.bias_h = np.random.normal(0.0, pow(1, -0.5), (hidden_nodes, 1))
        self.bias_o = np.random.normal(0.0, pow(1, -0.5), (output_nodes, 1))

        self.activation_function   = lambda x: scipy.special.expit(x)
        # lambda x: x * (x > 0)
        self.activation_function_d =  lambda x: self.activation_function(x) * (1 - self.activation_function(x))
        # lambda x: 1 * (x >= 0)
        self.sigmoid   = lambda x: scipy.special.expit(x)
        self.sigmoid_d = lambda y: y * (1 - y)

    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden = self.activation_function(np.dot(self.weights_ih, inputs) + self.bias_h)
        outputs = self.sigmoid(np.dot(self.weights_ho, hidden) + self.bias_o)
        return outputs.flatten().tolist()

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Feedforward
        hidden_inputs = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.activation_function(hidden_inputs)
        outputs = self.sigmoid(np.dot(self.weights_ho, hidden) + self.bias_o)

        # Calculate errors
        output_errors = targets - outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Calculate gradients
        output_gradient = output_errors * self.sigmoid_d(outputs) * self.learning_rate
        hidden_gradient = hidden_errors * self.activation_function_d(hidden_inputs) * self.learning_rate

        # Calculate deltas
        output_deltas = np.dot(output_gradient, hidden.T)
        hidden_deltas = np.dot(hidden_gradient, inputs.T)

        # Adjust weights and biases by deltas and gradients
        self.weights_ho += output_deltas
        self.weights_ih += hidden_deltas
        self.bias_o     += output_gradient
        self.bias_h     += hidden_gradient

    def save(self, path):
        data = {
            'weights_ih': self.weights_ih.tolist(),
            'weights_ho': self.weights_ho.tolist(),
            'bias_h': self.bias_h.tolist(),
            'bias_o': self.bias_o.tolist()
        }
        with open(path, 'w') as weights_file:
            json.dump(data, weights_file, separators=(',', ':'))

    def load(self, path):
        with open(path, 'r') as weights_file:
            data = json.load(weights_file)
            self.weights_ih = np.array(data['weights_ih'])
            self.weights_ho = np.array(data['weights_ho'])
            self.bias_h = np.array(data['bias_h'])
            self.bias_o = np.array(data['bias_o'])
