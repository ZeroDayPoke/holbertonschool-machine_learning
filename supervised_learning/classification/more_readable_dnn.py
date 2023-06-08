#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, input_size, layer_sizes, activation_function='sigmoid'):
        """
        Constructor for initializing a DeepNeuralNetwork.

        input_size : int
            The number of input features.
        layer_sizes : list
            The number of neurons in each layer, represented as a list.
        activation_function : str
            The activation function to be used. Must be either 'sigmoid' or 'tanh'. Default is 'sigmoid'.
        """
        # Check and assign the activation function.
        if activation_function not in ['sigmoid', 'tanh']:
            raise ValueError("activation_function must be 'sigmoid' or 'tanh'")
        self.__activation_function = activation_function

        # Check and assign the input size.
        if not isinstance(input_size, int):
            raise TypeError("input_size must be an integer")
        if input_size < 1:
            raise ValueError("input_size must be a positive integer")

        # Check and assign the layer sizes.
        if not isinstance(layer_sizes, list) or not layer_sizes:
            raise TypeError("layer_sizes must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layer_sizes)):
            raise TypeError("layer_sizes must be a list of positive integers")

        self.__layer_count = len(layer_sizes)
        self.__intermediate_values = {}
        self.__weights_and_biases = {}

        # Initialize the weights and biases for each layer.
        for i in range(self.__layer_count):
            self.__weights_and_biases['W' + str(i+1)] = np.random.randn(
                layer_sizes[i], input_size) * np.sqrt(2/input_size)
            self.__weights_and_biases['b' + str(i+1)] = np.zeros((layer_sizes[i], 1))
            input_size = layer_sizes[i]

    @property
    def layer_count(self):
        """Returns the total number of layers in the neural network."""
        return self.__layer_count

    @property
    def intermediate_values(self):
        """Returns the intermediate values (activations and inputs) calculated in the forward propagation."""
        return self.__intermediate_values

    @property
    def weights_and_biases(self):
        """Returns the weights and biases of the neural network."""
        return self.__weights_and_biases

    @property
    def activation_function(self):
        """Returns the activation function used in the neural network."""
        return self.__activation_function

    def forward_propagation(self, input_data):
        """
        Performs the forward propagation step.

        input_data : numpy.ndarray
            The input data.

        Returns
        -------
        A tuple containing:
            - The output of the last layer after forward propagation.
            - A dictionary containing the activations of each layer.
        """
        self.__intermediate_values['A0'] = input_data

        # Loop over each layer and calculate the linear combination and activation.
        for i in range(1, self.__layer_count + 1):
            linear_combination = np.dot(self.__weights_and_biases['W' + str(i)],
                                        self.__intermediate_values['A' + str(i - 1)]) +\
                                        self.__weights_and_biases['b' + str(i)]
            
            # Apply the softmax activation function if this is the output layer.
            if i == self.__layer_count:
                self.__intermediate_values['A' + str(i)] = self.softmax(linear_combination)
            else:
                if self.__activation_function == 'sigmoid':
                    self.__intermediate_values['A' + str(i)] = self.sigmoid(linear_combination)
                elif self.__activation_function == 'tanh':
                    self.__intermediate_values['A' + str(i)] = np.tanh(linear_combination)

        # Return the output of the last layer and the cache.
        return self.__intermediate_values['A' + str(self.__layer_count)], self.__intermediate_values

    def compute_cost(self, true_labels, predicted_labels):
        """
        Computes the cost of the prediction using the cross-entropy loss function.

        true_labels : numpy.ndarray
            The true labels of the data.
        predicted_labels : numpy.ndarray
            The predicted labels by the neural network.

        Returns
        -------
        The cost of the prediction.
        """
        num_examples = true_labels.shape[1]
        cost = -1 / num_examples * np.sum(true_labels * np.log(predicted_labels))
        return cost

    def evaluate(self, input_data, true_labels):
        """
        Evaluates the predictions made by the neural network.

        input_data : numpy.ndarray
            The input data.
        true_labels : numpy.ndarray
            The true labels of the data.

        Returns
        -------
        A tuple containing:
            - The prediction made by the neural network.
            - The cost of the prediction.
        """
        predicted_labels, _ = self.forward_propagation(input_data)
        cost = self.compute_cost(true_labels, predicted_labels)
        predictions = np.where(predicted_labels == np.amax(predicted_labels, axis=0), 1, 0)
        return predictions, cost

    def perform_gradient_descent(self, true_labels, intermediate_values, learning_rate=0.05):
        """
        Performs the gradient descent optimization step.

        true_labels : numpy.ndarray
            The true labels of the data.
        intermediate_values : dict
            The intermediate values (activations and inputs) calculated in the forward propagation.
        learning_rate : float
            The learning rate for the gradient descent optimization.

        """
        num_examples = true_labels.shape[1]

        # Loop over each layer in reverse order and calculate the gradients and update the weights and biases.
        for i in reversed(range(1, self.__layer_count + 1)):
            activation_values = intermediate_values['A' + str(i)]
            previous_activation_values = intermediate_values['A' + str(i - 1)]
            weights = self.__weights_and_biases['W' + str(i)]

            if i == self.__layer_count:
                derivative = activation_values - true_labels
            else:
                if self.__activation_function == 'sigmoid':
                    derivative = derivative_next * self.sigmoid_derivative(activation_values)
                elif self.__activation_function == 'tanh':
                    derivative = derivative_next * (1 - activation_values**2)  # derivative of tanh

            weights_derivative = np.dot(derivative, previous_activation_values.T) / num_examples
            bias_derivative = np.sum(derivative, axis=1, keepdims=True) / num_examples

            if i > 1:
                derivative_next = np.dot(weights.T, derivative)

            # Update the weights and biases.
            self.__weights_and_biases['W' + str(i)] -= learning_rate * weights_derivative
            self.__weights_and_biases['b' + str(i)] -= learning_rate * bias_derivative

    def train_network(self, input_data, true_labels, iterations=5000, learning_rate=0.05, verbose=True, plot_graph=True, step=100):
        """
        Trains the neural network.

        input_data : numpy.ndarray
            The input data.
        true_labels : numpy.ndarray
            The true labels of the data.
        iterations : int
            The number of iterations to train the network. Default is 5000.
        learning_rate : float
            The learning rate for the gradient descent optimization. Default is 0.05.
        verbose : bool
            Whether to print the cost after each iteration. Default is True.
        plot_graph : bool
            Whether to plot the cost after each iteration. Default is True.
        step : int
            The step size for printing and plotting the cost. Default is 100.

        Returns
        -------
        A tuple containing:
            - The prediction made by the neural network after training.
            - The cost of the prediction.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")
        if learning_rate < 0:
            raise ValueError("learning_rate must be positive")

        if verbose or plot_graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_values = []

        for i in range(iterations):
            predicted_labels, intermediate_values = self.forward_propagation(input_data)
            self.perform_gradient_descent(true_labels, intermediate_values, learning_rate)

            if i % step == 0 or i == iterations:
                cost = self.compute_cost(true_labels, predicted_labels)
                cost_values.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

        if plot_graph:
            plt.plot(np.arange(0, iterations + 1, step), cost_values)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(input_data, true_labels)

    def save_network(self, filename='uwu-saved.pkl'):
        """
        Saves the current state of the neural network to a file.

        filename : str
            The name of the file. Default is 'uwu-saved.pkl'.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_network(filename):
        """
        Loads a saved neural network from a file.

        filename : str
            The name of the file.

        Returns
        -------
        The loaded neural network.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    @staticmethod
    def sigmoid(linear_combination):
        """
        Applies the sigmoid activation function.

        linear_combination : numpy.ndarray
            The linear combination of inputs.

        Returns
        -------
        The output after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-linear_combination))

    @staticmethod
    def sigmoid_derivative(activation_value):
        """
        Calculates the derivative of the sigmoid function.

        activation_value : numpy.ndarray
            The output of the sigmoid activation function.

        Returns
        -------
        The derivative of the sigmoid function.
        """
        return activation_value * (1 - activation_value)

    @staticmethod
    def softmax(linear_combination):
        """
        Applies the softmax activation function.

        linear_combination : numpy.ndarray
            The linear combination of inputs.

        Returns
        -------
        The output after applying the softmax activation function.
        """
        exp = np.exp(linear_combination - np.max(linear_combination))
        return exp / exp.sum(axis=0, keepdims=True)

    @staticmethod
    def one_hot_encoder(labels, num_classes):
        """
        One-hot encodes the given labels.

        labels : numpy.ndarray
            The labels to be one-hot encoded.
        num_classes : int
            The total number of unique classes.

        Returns
        -------
        The one-hot encoded labels.
        """
        return np.eye(num_classes)[labels]

    @staticmethod
    def one_hot_decoder(one_hot_labels):
        """
        Decodes the one-hot encoded labels.

        one_hot_labels : numpy.ndarray
            The one-hot encoded labels.

        Returns
        -------
        The decoded labels.
        """
        return np.argmax(one_hot_labels, axis=1)
