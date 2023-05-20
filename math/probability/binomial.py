#!/usr/bin/env python3
""" Binomial distribution """


class Binomial:
    """
    Binomial distribution
        Args:
            data (list): List of the data to be
            used to estimate the distribution
            n (int): Number of Bernoulli trials
            p (float): Probability of a “success”
        Raises:
            ValueError: If n is not a positive value
            ValueError: If p is not a valid probability
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
    """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n, self.p = n, p
            if self.n <= 0:
                raise ValueError('n must be a positive value')
            if self.p <= 0 or self.p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((xi - mean) ** 2 for xi in data) / len(data)
            self.p = 1 - variance / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n
