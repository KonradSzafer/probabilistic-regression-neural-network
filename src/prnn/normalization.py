import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional


def MinMaxNormalizer(x: Tensor) -> Tensor:
    y = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return y


def ProbabilityNormalizer(x: Tensor) -> Tensor:
    y = x / torch.sum(x)
    return y


class Linear(nn.Module):
    def __init__(self) -> None:
        super(Linear, self).__init__()

    def __call__(self, x) -> Tensor:
        y = MinMaxNormalizer(x)
        y = ProbabilityNormalizer(y)
        return y


class ExpSoftmax(nn.Module):
    def __init__(
            self,
            dim: Optional[int] = None,
            factor: Optional[float] = 0.5
        ) -> None:
        super(ExpSoftmax, self).__init__()
        self.dim = dim
        self.factor = factor

    def __call__(self, x) -> Tensor:
        y = self.factor * torch.exp(x)
        y = nn.Softmax(dim=self.dim)(y)
        return y


class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def __call__(self, x) -> Tensor:
        y = nn.Sigmoid()(x)
        y = ProbabilityNormalizer(y)
        return y


class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def __call__(self, x) -> Tensor:
        y = nn.ReLU()(x)
        y = ProbabilityNormalizer(y)
        return y


class LeakyReLU(nn.Module):
    def __init__(
            self,
            negative_slope: Optional[float] = 0.1
        ) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def __call__(self, x) -> Tensor:
        y = nn.LeakyReLU(self.negative_slope)(x)
        y = torch.add(y, -torch.min(y))
        y = ProbabilityNormalizer(y)
        return y


def plot_functions(x: Tensor) -> None:
    plt.plot(x, Linear()(x), label='Linear')
    plt.plot(x, nn.Softmax(dim=-1)(x), label='Softmax')
    plt.plot(x, ExpSoftmax(dim=-1, factor=0.1)(x), label='Exp Softmax 0.1')
    plt.plot(x, ExpSoftmax(dim=-1, factor=0.3)(x), label='Exp Softmax 0.3')
    plt.plot(x, ExpSoftmax(dim=-1, factor=0.5)(x), label='Exp Softmax 0.5')
    plt.plot(x, Sigmoid()(x), label='Sigmoid')
    plt.plot(x, LeakyReLU(negative_slope=0.1)(x), label='Leaky ReLU 0.1')
    plt.plot(x, ReLU()(x), label='ReLU')
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
