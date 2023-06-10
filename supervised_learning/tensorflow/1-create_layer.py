#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function to create a layer for the neural network.

    Arguments:
    prev: tensor, output of the previous layer
    n: int, number of nodes in the layer to create
    activation: activation function that the layer should use

    Returns:
    layer: tensor, output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer_name = "layer" + activation.__name__

    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name=layer_name)

    return layer
