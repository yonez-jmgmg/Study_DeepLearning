import numpy as np
import matplotlib.pyplot as plt


def step_function_1input(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x, y, 'o-')
plt.show()

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
plt.plot(x, y, 'o-')
plt.show()

x = np.arange(-5, 5, 0.1)
y = relu(x)
plt.plot(x, y, 'o-')
plt.show()
