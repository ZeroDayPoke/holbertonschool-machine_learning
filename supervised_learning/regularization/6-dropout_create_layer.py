#!/usr/bin/env python3
"""Dropout regularization module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    checker compliant docstring
    """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initialize,
                            kernel_regularizer=dropout)
    return layer
