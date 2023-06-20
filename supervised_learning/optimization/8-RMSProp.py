#!/usr/bin/env python3
"""RMSPropUpgraded"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm.

    Args:
        loss (tf.Tensor): The loss of the network.
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.

    Returns:
        tf.Operation: The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
