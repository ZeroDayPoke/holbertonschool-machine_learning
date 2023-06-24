#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """
    checker hates docstrings
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Sum of each column (predicted positives) in the confusion matrix
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = True Positives / Predicted Positives
    precision = true_positives / predicted_positives

    return precision
