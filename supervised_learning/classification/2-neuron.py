#!/usr/bin/env python3
"""Class Neuron"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be a integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
