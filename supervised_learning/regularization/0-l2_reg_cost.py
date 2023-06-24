#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    docstrings to be added post checker
    """
    # Calculate the L2 regularization term
    l2_reg_term = 0
    for i in range(1, L + 1):
        l2_reg_term += np.sum(np.square(weights['W' + str(i)]))

    # Add the L2 regularization term to the original cost
    l2_cost = cost + lambtha / (2 * m) * l2_reg_term

    return l2_cost
