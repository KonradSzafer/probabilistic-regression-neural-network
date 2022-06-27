import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import torch
from torch import nn
from typing import List
from prnn.loss_functions import dist_loss
from prnn.normalization import *

Tensor = torch.Tensor
torch.autograd.set_detect_anomaly(True)

normalization_functions = {
    'softmax': nn.Softmax(dim=-1),
    'exp_softmax': ExpSoftmax(dim=-1, factor=0.1),
    'logarithmic': Logarithmic(),
    'sigmoid': Sigmoid(),
    'linear': Linear(),
    'relu': ReLU(),
    'leaky_relu': LeakyReLU(negative_slope=0.1)
}

class PRNN(nn.Module):

    def __init__(
            self,
            input_size: int,
            min_value: float=0,
            max_value: float=1,
            latent_resolution: int=10,
            intervals_precision: int=3
        ):
        super(PRNN, self).__init__()

        assert input_size > 0, \
            'input size must be greater than 0'
        assert max_value > min_value, \
            'max value must be greater than min value'
        assert latent_resolution > 1, \
            'latent resolution must be greater than 1'
        assert intervals_precision > 0, \
            'intervals precision must be greater than 0'

        self.min_value = min_value
        self.max_value = max_value
        self.latent_res = latent_resolution
        self.intervals_precision = intervals_precision

        # intervals for NN and ploting
        self.intervals = np.linspace(
            self.min_value,
            self.max_value,
            num=self.latent_res+1
        )
        self.intervals_val_dict = {}
        self.intervals_str_dict = {}
        for i in range(len(self.intervals)-1):
            self.intervals_val_dict[i] = [
                np.round(self.intervals[i], self.intervals_precision),
                np.round(self.intervals[i+1], self.intervals_precision)
            ]
            self.intervals_str_dict[i] = '<' + \
                str(f'{self.intervals[i]:.{self.intervals_precision}f}') + ', ' + \
                str(f'{self.intervals[i+1]:.{self.intervals_precision}f}') + ')'

        # neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(100, latent_resolution),
            # nn.Softmax(dim=1)
        )


    def forward(self, x):
        output = self.model(x)
        return output


    def print_intervals(self) -> None:
        print('class idx | min | max')
        for i, interval in self.intervals_str_dict.items():
            print('%d:' % i, interval)


    def digitize(self, input: Tensor) -> Tensor:
        # type conversion
        if torch.is_tensor(input):
            input = input.detach().cpu().numpy()
        elif type(input) is List:
            input = np.array(input)
        # digitalize
        output = np.digitize(
            input,
            self.intervals,
            right=False
        )
        # start class labels from 0
        output -= 1
        # torch conversion
        output = torch.from_numpy(output)
        output = output.type(torch.FloatTensor)
        return output


    def predict_sample(
            self,
            input: Tensor,
            normalize: bool=True,
            plot_distribution: bool=False
        ) -> Tensor:

        output = self.model(input.unsqueeze(0))
        _, label = torch.max(output, 1)
        label = label.item()
        interval = self.intervals_val_dict[label]
        return output, label, interval


    def plot_latent_distribution(
            self,
            output: Tensor,
            normalization: str=None,
            title: str='',
            filename: str='',
            dpi: int=200
        ) -> None:

        output = output.detach().squeeze(0)
        if normalization:
            output = normalization_functions[normalization](output)
        output = output.cpu().numpy()
        plt.bar(np.arange(len(output)), output)
        plt.title(title)
        plt.xticks(np.arange(len(output)), self.intervals_str_dict.values(), rotation=45);
        plt.xlabel('Interval')
        plt.ylabel('Probability')
        if filename:
            plt.savefig(filename, dpi=dpi)
        plt.show()


    def estimate(self, input: Tensor, method='default') -> Tensor:

        # check dimensions
        tensors_count, latent_size = input.shape
        assert \
            latent_size == self.latent_res, \
            f'incopatible latient size, expected \
            {self.latent_res} got {latent_size}'

        # output tensors
        value = torch.zeros([tensors_count, 1], dtype=torch.float16)
        probability = torch.zeros([tensors_count, 1], dtype=torch.float16)

        # normalization
        input = nn.Softmax(dim=-1)(input)

        for i in range(tensors_count):
            output: float = 0
            for j in range(latent_size):
                interval_mid = mean(self.intervals_val_dict[j])
                interval_prob = input[i,j].item()
                output += interval_mid * interval_prob
            value[i,0] = output
            probability[i,0], _ = torch.max(input[i], dim=-1)
        return value, probability
