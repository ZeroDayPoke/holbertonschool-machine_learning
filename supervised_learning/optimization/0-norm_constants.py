#!/usr/bin/env python3
""" Normalization Constants """
import numpy as np

def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix

    Arguments:
    X -- numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features

    Returns:
    mean and standard deviation of each feature, respectively
    """
    feature_mean = np.mean(X, axis=0)
    feature_std_dev = np.std(X, axis=0)
    return feature_mean, feature_std_dev
