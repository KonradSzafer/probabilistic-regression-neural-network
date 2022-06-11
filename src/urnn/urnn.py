import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import torch
from torch import nn
from typing import List

Tensor = torch.Tensor
torch.autograd.set_detect_anomaly(True)


class URNN(nn.Module):

    def __init__(
            self,
            input_size: int,
            min_value: float=0,
            max_value: float=1,
            latent_resolution: int=10,
            plot_bin_precision: int=3
        ):
        super(URNN, self).__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.latent_res = latent_resolution
        self.plot_bin_precision = plot_bin_precision

        # bins for NN and ploting
        self.bins = np.linspace(
            self.min_value,
            self.max_value,
            num=self.latent_res+1
        )
        self.bins_val_dict = {}
        self.bins_str_dict = {}
        for i in range(len(self.bins)-1):
            self.bins_val_dict[i] = [
                np.round(self.bins[i], self.plot_bin_precision),
                np.round(self.bins[i+1], self.plot_bin_precision)
            ]
            self.bins_str_dict[i] = '<' + \
                str(f'{self.bins[i]:.{self.plot_bin_precision}f}') + ', ' + \
                str(f'{self.bins[i+1]:.{self.plot_bin_precision}f}') + ')'

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


    def print_bins(self) -> None:
        print('class idx | bin min | bin max')
        for i, bins in self.bins_str_dict.items():
            print('%d:' % i, bins)


    def digitize(self, input: Tensor) -> Tensor:
        # type conversion
        if torch.is_tensor(input):
            input = input.detach().cpu().numpy()
        elif type(input) is List:
            input = np.array(input)
        # digitalize
        output = np.digitize(
            input,
            self.bins,
            right=False
        )
        # classes starting from 0
        output -= 1
        output = torch.from_numpy(output)
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
        bin = self.bins_val_dict[label]
        return output, label, bin


    def plot_latent_distribution(self, output: Tensor, normalize: bool=True) -> None:
        output = output.detach().squeeze(0)
        if normalize:
            output = nn.Softmax(dim=-1)(output)
        output = output.cpu().numpy()
        plt.bar(np.arange(len(output)), output)
        plt.xticks(np.arange(len(output)), self.bins_str_dict.values(), rotation=45);
        plt.xlabel('Bin index')
        plt.ylabel('Probability')
        plt.show()


    def estimate(self, input: Tensor, method='default') -> Tensor:

        # check dimensions
        tensors_count, latent_size = input.shape
        assert (
            latent_size == self.latent_res,
            f'incopatible latient size, expected {self.latent_res} got {latent_size}'
        )

        # output tensors
        value = torch.zeros([tensors_count, 1], dtype=torch.float16)
        probability = torch.zeros([tensors_count, 1], dtype=torch.float16)

        # normalization
        input = nn.Softmax(dim=-1)(input)

        for i in range(tensors_count):
            output: float = 0
            for j in range(latent_size):
                bin_mid = mean(self.bins_val_dict[j])
                bin_prob = input[i,j].item()
                output += bin_mid * bin_prob
            value[i,0] = output
            probability[i,0], _ = torch.max(input[i], dim=-1)
        return value, probability


if __name__ == '__main__':

    model = URNN(
        input_size=3,
        min_value=0.0,
        max_value=1+1e-2, # must be little above the range of values
        latent_resolution=5
    )
    model.print_bins()

    target = torch.FloatTensor([0.1, 0.3, 0.5, 0.7, 0.9])
    target = model.digitize(target)
    print('Target:', target)

    x = torch.FloatTensor([
        [0.7, 0.2, 0.1, 0.0, 0.0],
        [0.09, 0.5, 0.2, 0.2, 0.01],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.6, 0.2, 0.1, 0.09, 0.01],
        [0.6, 0.2, 0.1, 0.09, 0.01]
    ])

    from loss_functions import dist_loss
    output = dist_loss(x, target)
    print('Loss:', output)

    x = Tensor([0, 0.04, 0.07])
    output, label, bin = model.predict_sample(x)
    print(output, label, bin)

    value, probability = model.estimate(output)
    print('value:', value, 'probability:', probability)
