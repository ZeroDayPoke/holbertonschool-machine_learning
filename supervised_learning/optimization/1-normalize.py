#!/usr/bin/env python3
""" Normalization Constants """
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Arguments:
    X -- numpy.ndarray of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m -- numpy.ndarray of shape (nx,) that
        contains the mean of all features of X
    s -- numpy.ndarray of shape (nx,) that
        contains the standard deviation of all features of X

    Returns:
    The normalized X matrix
    """
    X_norm_np_mat = (X - m) / s
    return X_norm_np_mat
