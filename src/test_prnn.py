import torch
from torch import Tensor
from prnn import PRNN
from prnn.loss_functions import dist_loss, focal_loss


if __name__ == '__main__':

    model = PRNN(
        input_size=3,
        min_value=0.0,
        max_value=1+1e-2, # must be little above the range of values
        latent_resolution=5
    )
    model.print_intervals()

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

    output = dist_loss(x, target)
    print('Loss:', output)

    x = Tensor([0, 0.04, 0.07])
    output, label, interval = model.predict_sample(x)
    print(output, label, interval)

    value, probability = model.estimate(output)
    print('value:', value, 'probability:', probability)
