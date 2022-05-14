import numpy as np
import torch
from torch import nn
from typing import List

Tensor = torch.Tensor


class URNN(nn.Module):

    def __init__(
            self,
            input_size: int,
            min_value: float=0,
            max_value: float=1,
            latent_resolution: int=10
        ):
        super(URNN, self).__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.latent_res = latent_resolution

        self.bins = np.linspace(
            self.min_value,
            self.max_value,
            num=self.latent_res
        )

        # neural network
        self.regressor = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.Linear(800, latent_resolution),
            nn.Softmax()
        )


    # private?
    def digitalize(self, input: Tensor) -> Tensor:
        # conversion
        if torch.is_tensor(input):
            input = input.detach().cpu().numpy()
        if type(input) is List:
            input = np.array(input)
        # digitalize
        output = np.digitize(input, self.bins)
        output = torch.from_numpy(output)
        return output


    def loss(self, input: Tensor, target: Tensor) -> Tensor:

        if not len(input) == len(target):
            raise ValueError('Input and target size dont match')

        return None


if __name__ == '__main__':

    model = URNN(
        input_size=10,
        min_value=0.0,
        max_value=1.0,
        latent_resolution=5
    )

    target = torch.FloatTensor([0.1, 0.5, 0.02, 2.0])
    target = model.digitalize(target)
    print('Target:', target)

    x = torch.FloatTensor([
        [0.7, 0.2, 0.1, 0.0, 0.0],
        [0.09, 0.2, 0.5, 0.2, 0.01],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.6, 0.2, 0.1, 0.09, 0.01]
    ])
    out = model.loss(x, target)
    print(out)
