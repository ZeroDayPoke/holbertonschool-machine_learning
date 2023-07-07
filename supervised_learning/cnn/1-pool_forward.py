#!/usr/bin/env python3
"""
Module to perform a forward pass over a pooling layer in a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    parameters:
        - A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains output of previous layer
            * m: number of examples
            * h_prev: height of previous layer
            * w_prev: width of the previous layer
            * c_prev: number of channels in the previous layer
        - kernel_shape [tuple of (kh, kw)]:
            contains the size of the kernel for the pooling
            * kh: kernel height
            * kw: kernel width
        - stride [tuple of (sh, sw)]:
            contains strides for the pooling
            * sh: stride for the height
            * sw: stride for the width
        - mode [str]: 'max' or 'avg', indicating whether to perform
            maximum or average pooling, respectively

    returns:
        output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == "max":
                output[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == "avg":
                output[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return output
