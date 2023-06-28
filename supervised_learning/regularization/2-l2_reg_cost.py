#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    docstrings to be added post checker
    """
    # Add the L2 regularization losses to the original cost
    return cost + tf.losses.get_regularization_losses()
