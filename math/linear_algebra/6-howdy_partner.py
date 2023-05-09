#!/usr/bin/env python3
"""UWU Module"""


def cat_arrays(arr1, arr2):
    """
    Function that concatenates two arrays
    arr1 and arr2 are arrays containing ints/floats
    Return: a new array with arr1 and arr2 concatenated
    If arr1 and arr2 are not the same shape, return None
    Args:
        arr1: (list)    first array
        arr2: (list)    second array
    """
    catmat = arr1 + arr2
    return catmat
