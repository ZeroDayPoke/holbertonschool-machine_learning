#!usr/bin/env python3
"""Returns the sum of i^2 for i = 1 to n."""


def summation_i_squared(n):
    """Returns the sum of i^2 for i = 1 to n."""
    if not isinstance(n, int) or n < 1:
        return None
    return sum(i**2 for i in range(1, n+1))
