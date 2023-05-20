#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """
    Normal distribution
        Args:
          data (list): List of the data to be used to estimate the distribution
          mean (float): Expected mean of the data
          stddev (float): Expected standard deviation of the data
        Raises:
            ValueError: If stddev is not a positive value
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            sum_of_squares = sum((x - self.mean) ** 2 for x in data)
            self.stddev = (sum_of_squares / len(data)) ** 0.5

    def exponent(self, base, power):
        """Calculates base to the power of power"""
        return base ** power

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        num_e = 2.7182818285
        num_pi = 3.1415926536
        factor1 = 1 / (self.stddev * ((2 * num_pi) ** 0.5))
        factor2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        pdf_value = factor1 * self.exponent(num_e, -factor2)
        return pdf_value

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        z = self.z_score(x)
        return 0.5 * (1 + self.approx_erf(z / 2 ** 0.5))

    def approx_erf(self, x):
        """Approximates the error function erf(x)"""
        num_pi = 3.1415926536
        return (2 / (num_pi ** 0.5)) * (x
                                        - (x ** 3) / 3
                                        + (x ** 5) / 10
                                        - (x ** 7) / 42
                                        + (x ** 9) / 216)
