import matplotlib.pyplot as plt
import numpy as np


def step_function_1input(x):
    if x > 0:
        return 1
    else:
        return 0


# activate function for intermediate layer
def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# activate function for output layer
def identity_function(x):
    return x


# text初出のときはこっち。
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c)  # overflow対策
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


# test graph show
def test():
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

    x = np.arange(-5, 5, 0.1)
    y = identity_function(x)
    plt.plot(x, y, 'o-')
    plt.show()

    x = np.arange(-5, 5, 0.1)
    y = softmax(x)
    plt.plot(x, y, 'o-')
    plt.show()
