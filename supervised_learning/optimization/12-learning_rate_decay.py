#!/usr/bin/env python3
"""Learning Rate String"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate at which alpha will decay.
        global_step (int): The number of passes of gradient descent that have elapsed.
        decay_step (int): The number of passes of gradient descent that should occur before alpha is decayed further.

    Returns:
        tf.Operation: The learning rate decay operation.
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True,
    )
