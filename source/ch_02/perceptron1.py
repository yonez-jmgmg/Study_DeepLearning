import numpy as np
import matplotlib.pyplot as plt


def perceptron(theta, w1, w2, x1, x2):
    evaluation = w1 * x1 + w2 * x2
    if evaluation <= theta:
        return 0
    if evaluation > theta:
        return 1


print(perceptron(1, 0.5, 0.5, 0, 1))
print(perceptron(1, 0.5, 0.5, 3, 1))
print(perceptron(1, 0.5, 0.5, 0, 3))
