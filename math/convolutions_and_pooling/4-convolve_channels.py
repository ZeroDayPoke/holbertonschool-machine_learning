#!/usr/bin/env python3
"""Function that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = ((((h - 1) * sh) + kh - h) // 2) + 1
        pad_w = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    output_h = (h - kh + 2 * pad_h) // sh + 1
    output_w = (w - kw + 2 * pad_w) // sw + 1

    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')

    output = np.zeros((m, output_h, output_w))

    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = (padded_images[:, x *
                                             sh:x *
                                             sh +
                                             kh, y *
                                             sw:y *
                                             sw +
                                             kw, :] *
                               kernel).sum(axis=(1, 2, 3))

    return output
