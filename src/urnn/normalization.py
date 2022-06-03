import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def Softmax(x):
    y = nn.Softmax(dim=-1)(x)
    return y


def ExpSoftmax(x, base: float=1.0):
    x = base * torch.exp(x)
    y = nn.Softmax(dim=-1)(x)
    return y


def ReLU(x):
    y = F.normalize(x, dim=-1)
    y = nn.ReLU()(y)
    return y


if __name__ == '__main__':
    x = torch.linspace(-1.5, .95, steps=15)
    plt.plot(x, Softmax(x), label='Softmax')
    plt.plot(x, ExpSoftmax(x, base=1.0), label='Exp Softmax 1.0')
    plt.plot(x, ExpSoftmax(x, base=1.5), label='Exp Softmax 1.5')
    plt.plot(x, ReLU(x), label='ReLU')
    plt.legend()
    # plt.savefig('normalization_functions.png')
    plt.show()
