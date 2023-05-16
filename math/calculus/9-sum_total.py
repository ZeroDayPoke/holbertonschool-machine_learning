#!/usr/bin/env python3
"""Returns the sum of i^2 i = 1 to n."""


def summation_i_squared(n):
    """Returns the sum of i^2 i = 1 to n."""
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
