#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function to create the forward propagation graph for the neural network.

    Arguments:
    x: placeholder for the input data
    layer_sizes: list, containing the number of nodes
    in each layer of the network
    activations: list, containing the activation functions
    for each layer of the network

    Returns:
    y_pred: tensor, the prediction of the network in tensor form
    """

    layer_output = x
    for i in range(len(layer_sizes)):
        layer_output = create_layer(prev=layer_output,
                                    n=layer_sizes[i],
                                    activation=activations[i])

    y_pred = layer_output

    return y_pred
