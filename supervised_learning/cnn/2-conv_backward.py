#!/usr/bin/env python3
"""
Module to perform a backward pass over a convolutional layer in a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer of a neural network

    parameters:
        - dZ [numpy.ndarray of shape (m, h_new, w_new, c_new)]:
            contains the partial derivatives with respect to the unactivated
            output of the convolutional layer
            * m: number of examples
            * h_new: height of the output
            * w_new: width of the output
            * c_new: number of channels in the output
        - A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains output of the previous layer
            * h_prev: height of the previous layer
            * w_prev: width of the previous layer
            * c_prev: number of channels in the previous layer
        - W [numpy.ndarray of shape (kh, kw, c_prev, c_new)]:
            contains kernels for the convolution
            * kh: filter height
            * kw: filter width
        - b [numpy.ndarray of shape (1, 1, 1, c_new)]:
            contains biases applied to the convolution
        - padding [str]: 'same' or 'valid', indicating type of padding used
        - stride [tuple of (sh, sw)]:
            contains strides for the convolution
            * sh: stride for the height
            * sw: stride for the width

    returns:
        partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    dA_prev = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    start_h = h * sh
                    start_w = w * sw
                    end_h = start_h + kh
                    end_w = start_w + kw

                    a_slice = A_prev_pad[i, start_h:end_h, start_w:end_w, :]
                    da_prev_slice = dA_prev[i, start_h:end_h, start_w:end_w, :]

                    weights = W[:, :, :, c]
                    dz = dZ[i, h, w, c]

                    da_prev_slice += weights * dz
                    dA_prev[i, start_h:end_h, start_w:end_w, :] = da_prev_slice

                    dW[:, :, :, c] += a_slice * dz

    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
