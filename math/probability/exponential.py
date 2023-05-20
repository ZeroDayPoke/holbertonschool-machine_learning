#!/usr/bin/env python3
""" Exponential distribution """


class Exponential:
    """
    Exponential distribution
        Args:
            data (list): List of the data to be
            used to estimate the distribution
            lambtha (float): Expected number of
            occurrences in a given time frame
        Raises:
            ValueError: If lambtha is not positive value
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def exponent(self, base, power):
        """Calculates base to the power of power"""
        exp = 1.0
        for _ in range(int(power)):
            exp *= base
        return exp

    def pdf(self, x):
        """Calculate the probability density function"""
        if x < 0:
            return 0
        e_to_ne_lambtha_x = self.exponent(2.718281828459045, -self.lambtha * x)
        return self.lambtha * e_to_ne_lambtha_x
