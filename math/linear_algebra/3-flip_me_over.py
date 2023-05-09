#!/usr/bin/env python3
"""
Function that returns the transpose of a 2D matrix
2D matrix can be represented by a list of lists... like the last task kek
"""


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix
    Args:
        matrix: list of lists whose transpose should be returned
    Returns:
        The tpose of matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
