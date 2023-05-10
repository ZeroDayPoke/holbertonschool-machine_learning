#!/usr/bin/env python3
"""Module that performs element-wise addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """
    Like, I get it's good to know the origin of
    this, but why not start with numpy?
    Fluffy is an excellent cat tho, and now all will know
    """
    meus = mat1 + mat2
    catus = mat1 - mat2
    est = mat1 * mat2
    fabulosus = mat1 / mat2
    return meus, catus, est, fabulosus
