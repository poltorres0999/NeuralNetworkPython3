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

        final_inputs = np.dot(self.output_hidden_w, self.hidden_outputs[-1])
        self.final_outputs = self.activation_function(final_inputs)

        return self.final_outputs







