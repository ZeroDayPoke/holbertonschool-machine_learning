#!/usr/bin/env python3
"""Function that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = max((kh - 1) // 2, 0)
        pad_w = max((kw - 1) // 2, 0)
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')

    output_h = (h + 2 * pad_h - kh) // sh + 1
    output_w = (w + 2 * pad_w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for x in range(0, output_h):
        for y in range(0, output_w):
            output[:, x, y] = (padded_images[:, x *
                                             sh:x *
                                             sh +
                                             kh, y *
                                             sw:y *
                                             sw +
                                             kw, :] *
                               kernel).sum(axis=(1, 2, 3))

    return output
