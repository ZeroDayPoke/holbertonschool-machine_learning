#!/usr/bin/env python3
"""docstring"""


def moving_average(data, beta):
    """
    doc
    """
    v = 0
    moving_avgs = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        bias_correction = v / (1 - beta ** (i + 1))
        moving_avgs.append(bias_correction)
    return moving_avgs
