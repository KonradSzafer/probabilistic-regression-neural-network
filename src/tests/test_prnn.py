import unittest
import torch
from torch import Tensor
from prnn import PRNN
from prnn.loss_functions import dist_loss, focal_loss


class TestPRNN(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPRNN, self).__init__(*args, **kwargs)
        self.model = PRNN(
            input_size=3,
            min_value=0.0,
            max_value=1+1e-2, # must be little above the range of values
            latent_resolution=5
        )


    def test_digitalize(self):
        target = torch.FloatTensor([0.1, 0.3, 0.5, 0.7, 0.9])
        target = self.model.digitize(target)
        self.assertTrue(torch.equal(target, Tensor([0,1,2,3,4]).to(torch.int64)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
