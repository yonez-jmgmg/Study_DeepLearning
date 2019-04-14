import numpy as np


def simple_mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def simple_cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error(y: np.ndarray, t: np.ndarray, is_one_hot: bool = True):
    # y:input, t:teacher
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if is_one_hot:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
