import matplotlib.pyplot as plt
import math
import numpy as np


def mysign(x):
    if x == 0:
        return 1
    else:
        return x / abs(x)


plt.figure(1)

x = []
y = []
y1 = []

a = np.linspace(-10, 10, 100)
for i in a:
    x.append(i)
    # y.append(10 * math.sin(5 * i) + 7 * math.cos(4 * i))
    y.append(i / mysign(i))

plt.plot(x, y)
plt.show()
