#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """
    Normal distribution
        Args:
            data (list): List of the data to be
            used to estimate the distribution
            mean (float): Mean of the distribution
            stddev (float): Standard deviation of the distribution
        Raises:
            ValueError: If stddev is not positive value
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            sum_of_squares = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((sum_of_squares / len(data)) ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev
