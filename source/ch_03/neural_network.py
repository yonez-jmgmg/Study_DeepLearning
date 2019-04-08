import numpy as np

from source.ch_03.activation_functions import identity_function
from source.ch_03.activation_functions import sigmoid


def forward(W, b, x, f_act):
    a = np.dot(x, W) + b
    z = f_act(a)
    return z


def ini_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


network = ini_network()
x = np.array([1.0, 0.5])
y = forward(network['W1'], network['b1'], x, sigmoid)
y = forward(network['W2'], network['b2'], y, sigmoid)
y = forward(network['W3'], network['b3'], y, identity_function)
print(y)
