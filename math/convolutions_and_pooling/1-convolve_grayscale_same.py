#!/usr/bin/env python3
"""Function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images"""
    m, h, w, c = images.shape
    kh, kw = kernel.shape
    pad_h = ((h - 1) * 1 + kh - h) // 2
    pad_w = ((w - 1) * 1 + kw - w) // 2
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h + kh %
                          2), (pad_w, pad_w + kw %
                               2), (0, 0)), 'constant')
    output = np.zeros((m, h, w, c))
    kernel = np.expand_dims(kernel, -1)
    for x in range(h):
        for y in range(w):
            output[:, x, y, :] = (
                padded_images[:, x:x + kh, y:y + kw, :] * kernel).sum(axis=(1, 2))
    return output
