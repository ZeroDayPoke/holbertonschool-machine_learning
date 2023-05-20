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
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate p from data
            total_successes = sum(data)
            total_trials = len(data)
            self.p = total_successes / total_trials

            # Calculate n by rounding to the nearest integer
            self.n = round(total_trials * self.p)

            # Recalculate p using the rounded n
            self.p = total_successes / self.n
