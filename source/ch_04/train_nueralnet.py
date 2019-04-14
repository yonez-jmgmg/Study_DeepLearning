import numpy as np

from source.ch_04.layer_net import TwoLayerNet
from source.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 100
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print("Loop number: " + str(i))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.numerical_gradient(x_batch, t_batch)

    for key in ("W1", "W2", "b1", "b2"):
        network.params[key] -= learning_rate * grads[key]

    loss = network.loss(x_batch, t_batch)
    print("Loss: " + str(loss))
    train_loss_list.append(loss)
