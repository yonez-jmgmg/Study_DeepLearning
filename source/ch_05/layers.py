import numpy as np

from source.ch_03.activation_functions import softmax
from source.ch_04.loss_function import cross_entropy_error


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class MuLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# class Affine:
#     def __init__(self, W, b):
#         self.W = W
#         self.b = b
#         self.x = None
#         self.dW = None
#         self.db = None
#
#     def forward(self, x):
#         self.x = x
#         out = np.dot(x, self.W) + self.b
#
#         return out
#
#     def backward(self, dout):
#         dx = np.dot(dout, self.W.T)
#         self.dW = np.dot(self.x.T, dout)
#         self.db = np.sum(dout, axis=0)
#
#         return dx
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


def test():
    print('ADD Layer test')
    add = AddLayer()
    print(add.forward(3, 6))
    print(add.backward(1))

    print('MUL Layer test')
    mul = MuLayer()
    print(mul.forward(3, 6))
    print(mul.backward(1))

    print('Relu Layer test')
    re = ReluLayer()
    print(re.forward(np.array([2, 1, 0, -5, 4])))
    print(re.backward(np.array([1, 1, 1, 1, 1])))


test()
