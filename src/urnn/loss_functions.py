import numpy as pn
import torch
import torch.nn as nn
import torch.nn.functional as F


def dist_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    if not len(input) == len(target):
        raise ValueError('Input and target size must match')

    samples_count = input.shape[0]
    classes_count = input.shape[1]
    output = torch.zeros(samples_count, dtype=torch.float16) #, requires_grad=True)

    for i, target_idx in enumerate(target):
        # calculate loss for each target separately
        loss = 0
        for probability_idx in range(classes_count):
            probability_value = input[i, probability_idx]
            bin_dist = torch.abs(target_idx - probability_idx)
            loss += probability_value * bin_dist
        output[i] = loss
        # output.data[i] = loss
    return output


@torch.jit.script
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: float=2.0,
    # weight=None,
    reduction: str='none'
    ) -> torch.Tensor:

    ce_loss = F.cross_entropy(input, target, reduction='none') # weight=weight,
    p_t = torch.exp(-ce_loss)
    loss = (1 - p_t)**gamma * ce_loss
    if reduction =='mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


if __name__ == '__main__':

    input = torch.FloatTensor([[0.2, 0.3, 0.1, 0.4]])
    target = torch.FloatTensor([[1]])
    loss = dist_loss(input, target)
