#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function to create placeholders for input data
    and labels for the neural network.
    Arguments:
    nx: int, the number of feature columns in our data
    classes: int, the number of classes in our classifier

    Returns:
    x: placeholder for the input data to the neural network
    y: placeholder for the one-hot labels for the input data
    """

    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")

    return x, y
