#!/usr/bin/env python3
"""Function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = max((kh - 1) // 2, 0)
    pad_w = max((kw - 1) // 2, 0)
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
    output = np.zeros((m, h, w))
    for x in range(h):
        for y in range(w):
            output[:, x, y] = (
                padded_images[:, x:x + kh, y:y + kw] * kernel).sum(axis=(1, 2))
    return output
