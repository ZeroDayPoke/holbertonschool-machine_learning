#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Create the training operation for the network
    loss is the loss of the network’s prediction
    alpha is the learning rate
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)

    return train_op
