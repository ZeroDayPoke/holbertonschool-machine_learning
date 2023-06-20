#!/usr/bin/env python3
"""RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (np.ndarray): A numpy.ndarray containing the variable to be updated.
        grad (np.ndarray): A numpy.ndarray containing the gradient of var.
        s (np.ndarray): The previous second moment of var.

    Returns:
        np.ndarray: The updated variable.
        np.ndarray: The new moment.
    """
    S = beta2 * s + (1 - beta2) * grad ** 2
    var_updated = var - alpha * grad / (np.sqrt(S) + epsilon)
    return var_updated, S
