#!/usr/bin/env python3
"""Poisson distribution"""
import math


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

    def pmf(self, k):
        # Convert k to an integer
        k = int(k)

        # If k is negative, it's out of range
        if k < 0:
            return 0

        # Calculate λ^k (lambda to the power of k)
        lambtha_to_k = self.lambtha ** k

        # Calculate e^-λ (Euler's number to the power of -lambda)
        e_to_lambda = math.exp(-self.lambtha)

        # Calculate k factorial
        k_factorial = math.factorial(k)

        # Calculate and return the PMF
        return (lambtha_to_k * e_to_lambda) / k_factorial

    def cdf(self, k):
        # Convert k to an integer
        k = int(k)

        # If k is negative, it's out of range
        if k < 0:
            return 0

        # Calculate the CDF by summing the PMFs from 0 to k
        cdf_value = sum(self.pmf(i) for i in range(k + 1))

        return cdf_value
