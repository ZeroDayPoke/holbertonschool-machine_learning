#!/usr/bin/env python3
"""RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    doc too dangerous for checker
    """
    S = beta2 * s + (1 - beta2) * grad ** 2
    var_updated = var - alpha * grad / (np.sqrt(S) + epsilon)
    return var_updated, S
