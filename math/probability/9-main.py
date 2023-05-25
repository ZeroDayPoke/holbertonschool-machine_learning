#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()

n1 = Normal(data)
print('===== NORMAL ===== DATA GIVEN =====')
print('n1 cdf(90):', n1.cdf(90))
print('n1 mu:', n1.mean)
print('n1 sigma:', n1.stddev)
print('n1 pdf(90):', n1.pdf(90))

n2 = Normal(mean=70, stddev=10)
print('===== NORMAL ===== DATA NOT GIVEN =====')
print('n2 cdf(90):', n2.cdf(90))
print('n2 mu:', n2.mean)
print('n2 sigma:', n2.stddev)
print('n2 pdf(90):', n2.pdf(90))
