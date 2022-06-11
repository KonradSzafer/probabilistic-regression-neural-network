import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def MinMaxNormalizer(x):
    y = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return y


def ProbabilityNormalizer(x):
    y = x / torch.sum(x)
    return y


def LinearlyNormalized(x):
    y = MinMaxNormalizer(x)
    y = ProbabilityNormalizer(y)
    return y


def Softmax(x):
    y = nn.Softmax(dim=-1)(x)
    return y


def ExpSoftmax(x, factor: float=1.0):
    x = factor * torch.exp(x)
    y = nn.Softmax(dim=-1)(x)
    return y


def Sigmoid(x):
    y = nn.Sigmoid()(x)
    y = ProbabilityNormalizer(y)
    return y


def ReLU(x):
    y = nn.ReLU()(x)
    y = ProbabilityNormalizer(y)
    return y


def LeakyReLU(x, negative_slope=0.1):
    y = nn.LeakyReLU(negative_slope)(x)
    y = torch.add(y, -torch.min(y))
    y = ProbabilityNormalizer(y)
    return y


def plot_functions(x):
    plt.plot(x, LinearlyNormalized(x), label='Linearly Normalized')
    plt.plot(x, Softmax(x), label='Softmax')
    plt.plot(x, ExpSoftmax(x, factor=0.5), label='Exp Softmax 0.5')
    plt.plot(x, ExpSoftmax(x, factor=1.0), label='Exp Softmax 1.0')
    plt.plot(x, ExpSoftmax(x, factor=1.5), label='Exp Softmax 1.5')
    plt.plot(x, Sigmoid(x), label='Sigmoid')
    plt.plot(x, LeakyReLU(x, negative_slope=0.1), label='Leaky ReLU 0.1')
    plt.plot(x, ReLU(x), label='ReLU') # dont sum to 1.
    plt.legend()
    plt.xlabel('neural network output')
    plt.ylabel('probability')
    # plt.savefig('normalization_functions.png')
    plt.show()


if __name__ == '__main__':

    # x = torch.linspace(-1.5, .95, steps=15)
    # x = torch.linspace(-1.5, .95, steps=200)

    x = torch.linspace(-3.5, 1.5, steps=30)
    plot_functions(x)

    x = torch.linspace(-8.5, 2.5, steps=30)
    plot_functions(x)
