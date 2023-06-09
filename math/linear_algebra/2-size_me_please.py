#!/usr/bin/env python3
"""
Module that calculates the shape of a matrix

Return the shape as a list of integers

If the matrix is not a list of lists, return None... since is no a matrix XD
"""


def matrix_shape(matrix):
    """
    Args:
    matrix (list): A nested list representing the matrix

    Returns:
    list: A list of integers representing the dimensions of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
