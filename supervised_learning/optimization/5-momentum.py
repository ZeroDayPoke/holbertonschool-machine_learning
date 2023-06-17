#!/usr/bin/env python3
"""Moment"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """don't have time for this right now"""
    v = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v
    return var_updated, v
