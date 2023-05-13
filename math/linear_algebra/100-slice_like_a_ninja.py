#!/usr/bin/env python3
""" Slice Like A Ninja """
import numpy as np

def np_slice(matrix, axes={}):
    """ slices a matrix along specific axes """
    slices = [slice(None)] * matrix.ndim
    for axis, slice_range in axes.items():
        slices[axis] = slice(*slice_range)
    return matrix[tuple(slices)]
