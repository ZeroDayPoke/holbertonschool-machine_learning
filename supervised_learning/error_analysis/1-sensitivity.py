#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    checker hates docstrings
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Sum of each row (actual positives) in the confusion matrix
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity = True Positives / Actual Positives
    sensitivity = true_positives / actual_positives

    return sensitivity
