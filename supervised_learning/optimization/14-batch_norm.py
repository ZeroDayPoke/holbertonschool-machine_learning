#!/usr/bin/env python3
"""Batch Normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (tf.Operation): The activation function that should be used on the output of the layer.

    Returns:
        tf.Tensor: A tensor of the activated output for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    dense = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        Z, mean, var, offset=beta, scale=gamma, variance_epsilon=1e-8)
    return activation(Z_norm)
