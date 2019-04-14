import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient_1d(f, x: np.ndarray):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp
    return grad


def numerical_gradient_2d(f, x):
    if x.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def gradient_descent(f, x0, lr=0.01, step_num=100):
    x = x0
    for i in range(step_num):
        grad = numerical_diff(f, x)
        x = x - lr * grad

    return x


def test():
    def function_2(x):
        return x[0] ** 2 + x[1] ** 2

    print(numerical_gradient_1d(function_2, np.array([3.0, 4.0])))
