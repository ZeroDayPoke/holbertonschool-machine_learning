#!/usr/bin/env python3
"""Learning Rate String"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    doc too dangerous for checker
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True,
    )
