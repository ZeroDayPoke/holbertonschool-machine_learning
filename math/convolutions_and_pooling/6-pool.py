#!/usr/bin/env python3
"""Function that performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for x in range(0, output_h):
        for y in range(0, output_w):
            if mode == 'max':
                output[:, x, y, :] = np.max(
                    images[:, x * sh:x * sh + kh, y * sw:y * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                output[:, x, y, :] = np.mean(
                    images[:, x * sh:x * sh + kh, y * sw:y * sw + kw, :],
                    axis=(1, 2))

    return output
