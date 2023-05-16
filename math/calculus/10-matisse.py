#!/usr/bin/env python3
"""Returns the derivative of a polynomial."""


def poly_derivative(poly):
    """
    Returns the derivative of a polynomial.
    Basically the returned list will be the
    coefficients of the derivative of the polynomial.
    """
    if not all(isinstance(i, (int, float)) for i in poly):
        return None

    if len(poly) <= 1:
        return [0]

    return [i * poly[i] for i in range(1, len(poly))]
