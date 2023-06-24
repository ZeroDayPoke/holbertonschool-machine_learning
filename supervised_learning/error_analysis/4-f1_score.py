#!/usr/bin/env python3
"""F1 score"""


def f1_score(confusion):
    """
    checker hates docstrings
    """
    # Import the sensitivity and precision functions
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    # Calculate sensitivity and precision
    sens = sensitivity(confusion)
    prec = precision(confusion)

    # F1 score = 2 * (precision * sensitivity) / (precision + sensitivity)
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
