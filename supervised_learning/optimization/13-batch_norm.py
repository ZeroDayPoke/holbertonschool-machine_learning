#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Args:
        Z (np.ndarray): The numpy.ndarray of shape (m, n) that should be normalized.
            m is the number of data points.
            n is the number of features in Z.
        gamma (np.ndarray): The numpy.ndarray of shape (1, n) containing the scales used for batch normalization.
        beta (np.ndarray): The numpy.ndarray of shape (1, n) containing the offsets used for batch normalization.
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        np.ndarray: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_final = gamma * Z_norm + beta
    return Z_final
