#!/usr/bin/env python3
"""
stack me
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fig, ax = plt.subplots()
bottom = np.zeros(3)
for i in range(4):
    ax.bar(people, fruit[i], width=0.5, bottom=bottom, color=colors[i])
    bottom += fruit[i]
ax.set_title('Number of Fruit per Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_yticks(range(0, 81, 10))
ax.legend(fruit_names)
plt.show()
