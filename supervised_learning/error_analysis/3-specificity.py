#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """
    checker hates docstrings
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # False positives are the sum of each column minus the true positives
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False negatives are the sum of each row minus the true positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    """
    True negatives are the sum of all elements minus the sum of false
    positives and false negatives and true positives
    """
    true_negatives = np.sum(confusion) - \
        (false_positives + false_negatives + true_positives)

    # Specificity = True Negatives / (True Negatives + False Positives)
    specificity = true_negatives / (true_negatives + false_positives)

    return specificity
