#!/usr/bin/env python3
"""
It just adds arrays... providing they are the same size ofc
"""


def add_arrays(arr1, arr2):
    """
    Function that adds two arrays 'element-wise'
    Args:
        arr1: First array
        arr2: Second array
    Returns:
        A new array with the sum of the two arrays
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
