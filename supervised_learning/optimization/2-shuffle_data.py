#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Arguments:
    X -- first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y -- second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns:
    The shuffled X and Y matrices
    """
    assert len(X) == len(Y), 'uniformity error'
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]
