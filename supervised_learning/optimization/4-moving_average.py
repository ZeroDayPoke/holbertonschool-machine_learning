#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of data.
    """
    v = 0
    moving_avgs = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        bias_correction = v / (1 - beta ** (i + 1))
        moving_avgs.append(bias_correction)
    return moving_avgs
