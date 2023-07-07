#!/usr/bin/env python3
"""
Module to perform a backward pass over a pooling layer in a neural network
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of a neural network

    parameters:
        - dA [numpy.ndarray of shape (m, h_new, w_new, c_new)]:
            contains the partial derivatives with respect to the output
            of the pooling layer
            * m: number of examples
            * h_new: height of the output
            * w_new: width of the output
            * c_new: number of channels
        - A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains output of the previous layer
            * h_prev: height of the previous layer
            * w_prev: width of the previous layer
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
        partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    start_h = h * sh
                    start_w = w * sw
                    end_h = start_h + kh
                    end_w = start_w + kw

                    if mode == "max":
                        a_prev_slice = A_prev[i,
                                              start_h:end_h, start_w:end_w, c]
                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i, start_h:end_h, start_w:end_w,
                                c] += mask * dA[i, h, w, c]
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        average = da / (kh * kw)
                        dA_prev[i, start_h:end_h, start_w:end_w,
                                c] += np.ones((kh, kw)) * average

    return dA_prev
