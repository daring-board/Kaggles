import os
import numpy as np
import matplotlib.pyplot as plt

print('start read')
data = [row for row in open('../data/cities.csv', 'r')]
data = data[1:]
cities = [row.strip().split(',') for row in data]
cities = {int(row[0]): (float(row[1]), float(row[2])) for row in cities}

print('end read')

x, y = [], []
for key in cities.keys():
    x.append(cities[key][0])
    y.append(cities[key][1])

plt.scatter( x, y, s=1)
plt.grid()
plt.show()
