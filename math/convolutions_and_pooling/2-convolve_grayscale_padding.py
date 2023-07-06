#!/usr/bin/env python3
"""
Function that performs a convolution on
grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on
    grayscale images with custom padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        'constant')
    output_h = h + 2 * pad_h - kh + 1
    output_w = w + 2 * pad_w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = (
                padded_images[:, x:x + kh, y:y + kw] * kernel).sum(axis=(1, 2))
    return output
