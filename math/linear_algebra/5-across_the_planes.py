#!/usr/bin/env python3
"""UWU Module"""


def add_matrices2D(mat1, mat2):
    """
    Function that adds two matrices
    mat1 and mat2 are 2D matrices containing ints/floats
    Return: a new matrix with the sum of mat1 and mat2
    If mat1 and mat2 are not the same shape, return None
    Args:
        mat1: (list)    first matrix
        mat2: (list)    second matrix
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    mat3 = [[0 for uwu in range(len(mat1[0]))] for uwu in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            mat3[i][j] = mat1[i][j] + mat2[i][j]
    return mat3
