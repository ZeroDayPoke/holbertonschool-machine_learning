#!/usr/bin/env python3
""" 17. Integrate """


def poly_integral(poly, C=0):
    """
    Returns the coefficients in list form
    representative of the integral of a polynomial.
    C is the integration constant (default 0)
    """
    if not isinstance(poly, list):
        return None

    if not all(isinstance(i, (int, float)) for i in poly):
        return None

    if not isinstance(C, int):
        return None
    
    if len(poly) == 0:
        return None

    integral = [C]
    for i in range(len(poly)):
        coefficient = poly[i] / (i + 1)
        if coefficient.is_integer():
            coefficient = int(coefficient)
        integral.append(coefficient)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
