#!/usr/bin/env python3
"""Adam wants more docs lol"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (np.ndarray): The variable to be updated.
        grad (np.ndarray): The gradient of var.
        v (np.ndarray): The previous first moment of var.
        s (np.ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.

    Returns:
        np.ndarray: The updated variable.
        np.ndarray: The new first moment.
        np.ndarray: The new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
