#!/usr/bin/env python3
"""
Pirate Module
is it good practice? no
is it fun? yes
have I been listening to too many sea shanties? also yes
"""


def mat_mul(mat1, mat2):
    """
    Matrix multiplication, but pirate themed this time!
    Args:
        mat1 (list): first ship
        mat2 (list): second ship
    Returns:
        list: the treasure totes obvs
        I mean honestly what else would it be
    """
    if len(mat1[0]) != len(mat2):
        return None
    treasure = [[0 for clue in range(len(mat2[0]))] for clue in range(len(mat1))]
    for island in range(len(mat1)):
        for x_marks_the_spot in range(len(mat2[0])):
            for dubloon in range(len(mat2)):
                treasure[island][x_marks_the_spot] += mat1[island][dubloon] * mat2[dubloon][x_marks_the_spot]
    return treasure
