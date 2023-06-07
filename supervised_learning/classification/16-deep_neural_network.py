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
        for i in layers:
            if not isinstance(i, int) or i < 1:
                raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer_dims = [nx] + layers
        for i in range(1, self.__L + 1):
            self.__weights['W'+str(i)] = np.random.randn(
                layer_dims[i], layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
            self.__weights['b'+str(i)] = np.zeros((layer_dims[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
