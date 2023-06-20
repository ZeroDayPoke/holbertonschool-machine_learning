# Probability - Machine Learning

This directory contains Python scripts that implement various probability distributions. The implemented distributions include Poisson, Exponential, Normal, and Binomial classes. Part of Holberton Machine Learning Specialization.

## File Descriptions

### [binomial.py](https://github.com/ZeroDayPoke/holbertonschool-machine_learning/tree/master/math/probability/binomial.py)

This script contains a Python class `Binomial` that represents a binomial distribution. The class takes a list of data or the number of Bernoulli trials and their probability of success. It includes methods to calculate the mean, variance, probability mass function (PMF), and cumulative distribution function (CDF) of the binomial distribution.

### [exponential.py](https://github.com/ZeroDayPoke/holbertonschool-machine_learning/tree/master/math/probability/exponential.py)

This Python script contains a class Exponential that represents an exponential distribution. The class takes either a list of data or a lambda value to estimate the distribution. It raises a ValueError if the lambda is not a positive value or if the data does not contain at least two data points. It also raises a TypeError if the data is not a list. The class includes methods to calculate the value of the PDF and CDF for a given time period.

### [normal.py](https://github.com/ZeroDayPoke/holbertonschool-machine_learning/tree/master/math/probability/normal.py)

This script contains a Python class Normal that represents a normal distribution. The class takes a list of data or the mean and standard deviation of the distribution. It includes methods to calculate the z-score, x-value, probability density function (PDF), and cumulative distribution function (CDF) of the normal distribution.

### [poisson.py](https://github.com/ZeroDayPoke/holbertonschool-machine_learning/tree/master/math/probability/poisson.py)

This script contains a Python class Poisson that represents a Poisson distribution. The class takes a list of data or the expected number of occurrences in a given time frame (lambda). It includes methods to calculate the probability mass function (PMF) and cumulative distribution function (CDF) of the Poisson distribution.

## Author

Chris Stamper - [ZeroDayPoke](https://github.com/ZeroDayPoke)
