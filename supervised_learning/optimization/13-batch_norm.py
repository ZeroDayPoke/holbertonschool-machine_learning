#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    doc too dangerous for checker
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_final = gamma * Z_norm + beta
    return Z_final
