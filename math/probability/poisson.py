#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """
    Poisson distribution
        Args:
            data (list): List of the data to be
            used to estimate the distribution
            lambtha (int/float): Expected number of
            occurences in a given time frame
        Raises:
            ValueError: If lambtha is not positive value
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
        Methods:
            pmf(k): Calculates the value of the PMF
            for a given number of “successes”
            cdf(k): Calculates the value of the CDF
            for a given number of “successes”
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
            self.lambtha = sum(data) / len(data)

    def exponent(self, base, power):
        """Calculates base to the power of power"""
        return base ** power

    def factorial(self, k):
        """Calculates factorial of k"""
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return fact

    def pmf(self, k):
        """Calculate the probability mass function"""
        k = int(k)
        if k < 0:
            return 0

        # Calculate each component of the PMF formula
        lambtha_to_k = self.exponent(self.lambtha, k)
        # hard coding e^-lambtha 2.71828 since no imports
        e_to_neg_lambda = self.exponent(2.71828, -self.lambtha)
        k_factorial = self.factorial(k)

        # Return PMF value
        return (lambtha_to_k * e_to_neg_lambda) / k_factorial

    def cdf(self, k):
        """Calculate the cumulative distribution function"""
        k = int(k)
        if k < 0:
            return 0

        # Calculate the CDF by summing the PMFs from 0 to k
        cdf_value = sum(self.pmf(i) for i in range(k + 1))

        return cdf_value
