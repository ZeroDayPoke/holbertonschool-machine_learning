#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            self.__weights['W' + str(i+1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2/nx)
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            Zi = np.dot(self.__weights['W' + str(i)],
                        self.__cache['A' + str(i - 1)]) +\
                        self.__weights['b' + str(i)]
            self.__cache['A' + str(i)] = self.sigmoid(Zi)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    @staticmethod
    def sigmoid(Z):
        """Sigmoid Activation"""
        return 1 / (1 + np.exp(-Z))
