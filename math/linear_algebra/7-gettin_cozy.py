#!/usr/bin/env python3
"""UWU Module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices
    mat1 and mat2 are 2D matrices containing ints/floats
    axis is the axis to concatenate on
    Return: a new matrix with the concatenation
    If mat1 and mat2 are not the same shape, return None
    Args:
        mat1: (list)    first matrix
        mat2: (list)    second matrix
        axis: (int)     axis to concatenate on
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None
    if axis == 0:
        meowMat2 = mat1 + mat2
    elif axis == 1:
        meowMat2 = [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    return meowMat2
