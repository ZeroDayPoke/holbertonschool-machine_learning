#!/usr/bin/env python3
"""Class Neuron"""

import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
