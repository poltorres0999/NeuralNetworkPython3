import numpy as np
import scipy as sp


class NeuralNetwork:

    def __init__(self, input_nodes, output_nodes, hidden_layers_nodes, learning_rate):
        # Parameters of the neural network
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layer_nodes = hidden_layers_nodes
        self.learning_rate = learning_rate

        # Activation function(sigmoid)
        self.activation_function = lambda x: sp.special.expit(x)
        # Initialize random weight matrix for the input and first hidden layer
        self.input_hidden_w = np.random.normal(0.0, pow(self.hidden_layer_nodes[0], -0.5),
                                               (self.hidden_layer_nodes[0], self.input_nodes))
        # Initializes random weight matrix for all the hidden layers
        self.hidden_w = []
        for i in range(len(hidden_layers_nodes) - 1):
            self.hidden_w.append(np.random.normal(0.0, pow(self.hidden_layer_nodes[i + 1], -0.5),
                                                  (self.hidden_layer_nodes[i + 1], self.hidden_layer_nodes[i])))
        # Initialize random weight matrix for the input and first hidden layer
        self.output_hidden_w = np.random.normal(0.0, pow(self.output_nodes[0], -0.5),
                                                (self.output_nodes, self.hidden_layer_nodes[-1]))
        # Keeps the result of the forwarding part of the iteration
        self.first_hidden_outputs = 0
        self.hidden_outputs = []
        self.final_outputs = 0

    def query(self, init_values):

        if len(init_values) != len(self.input_nodes):
            raise ValueError("The number of input nodes and initial values are not equal")
        # Calculates the outputs of the first hidden layer
        first_hidden_inputs = np.dot(self.input_hidden_w, init_values)
        self.first_hidden_outputs = self.activation_function(first_hidden_inputs)

        # Calculates the outputs for each hidden layer
        for i in range(len(self.hidden_w)):
            if i == 0:
                self.hidden_outputs.append(self.activation_function(
                    np.dot(self.first_hidden_outputs, self.hidden_w[i])))
            else:
                self.hidden_outputs.append(self.activation_function(
                    (self.hidden_outputs[i-1], self.hidden_w[i])))

        # Calculates the final outputs using the last outputs of hidden layer
        final_inputs = np.dot(self.output_hidden_w, self.hidden_outputs[-1])
        self.final_outputs = self.activation_function(final_inputs)

        return self.final_outputs

    def train(self, init_values, expected_output_values):

        # Convert the input values and the expected outputs in 2d arrays
        init_values = np.array(init_values, ndmin=2).T
        expected_output_values = np.array(expected_output_values, ndmin=2).T

        # Uses the query function to calculate the final outputs to start wih error calculation and weight updating
        final_outputs = self.query(init_values)

        # Calculates the Output errors
        output_errors = expected_output_values - final_outputs

        # Actualizes the last weight matrix
        self.output_hidden_w += self.learning_rate * np.dot((output_errors * final_outputs * (1-0 * final_outputs)),
                                                            np.transpose(self.hidden_outputs[-1]))
        # Actualizes the hidden matrix weights
        current_error = 0
        for i in range(len(self.hidden_outputs), -1, -1):
            if i == len(self.hidden_outputs):
                current_error = np.dot(self.output_hidden_w.T, output_errors)
                self.hidden_w[i] += self.learning_rate * np.dot((current_error * self.hidden_outputs[i] *
                                                (1-0 * self.hidden_outputs[i])), np.transpose(self.hidden_outputs[i-1]))
            elif i == 0:
                self.hidden_w[i] += self.learning_rate * np.dot((current_error * self.hidden_outputs[i] *
                                             (1 - 0 * self.hidden_outputs[i])), np.transpose(self.first_hidden_outputs))
            else:
                current_error = np.dot(self.hidden_w[i].T, current_error)
                self.hidden_w[i] += self.learning_rate * np.dot((current_error * self.hidden_outputs[i] *
                                                (1-0 * self.hidden_outputs[i])), np.transpose(self.hidden_outputs[i-1]))

        # Actualizes the input_hidden matrix weight
        current_error = np.dot(self.input_hidden_w.T, current_error)
        self.input_hidden_w += self.learning_rate * np.dot((current_error * self.first_hidden_outputs *
                                                         (1 - 0 * self.first_hidden_outputs)),np.transpose(init_values))










