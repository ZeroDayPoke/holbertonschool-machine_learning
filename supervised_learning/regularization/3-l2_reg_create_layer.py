#!/usr/bin/env python3
"""L2 Regularization Cost module"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    checker compliant docstring
    """
    regularization = tf.contrib.layers.l2_regularizer(lambtha)
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_regularized_layer = tf.layers.Dense(
        n, activation=activation,
        kernel_regularizer=regularization, kernel_initializer=initialize
    )
    return l2_regularized_layer(prev)
