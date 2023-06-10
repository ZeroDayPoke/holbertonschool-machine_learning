#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculate the softmax cross-entropy loss of a prediction.
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
