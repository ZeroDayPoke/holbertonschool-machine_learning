#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf

def create_layer(prev, n, activation, i):
    """
    Function to create a layer for the neural network.

    Arguments:
    prev: tensor, output of the previous layer
    n: int, number of nodes in the layer to create
    activation: activation function that the layer should use
    i: index of the layer

    Returns:
    layer: tensor, output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    activation_name = activation.__name__ if activation is not None else "None"
    layer_name = "layer" + str(i) + "_" + activation_name

    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name=layer_name)

    return layer
