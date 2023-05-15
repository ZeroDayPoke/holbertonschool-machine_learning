#!/usr/bin/env python3
"""topography"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=z, cmap='viridis')
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')
cbar = plt.colorbar()
cbar.set_label('elevation (m)')
plt.show(block=True)
