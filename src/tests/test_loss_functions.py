import unittest
import torch
from torch import FloatTensor
from prnn.loss_functions import dist_loss


class TestLossFunctions(unittest.TestCase):

    def test_dist_loss(self):
        target = FloatTensor([
            0.1, 0.3, 0.5, 0.7, 0.9
        ])
        x = FloatTensor([
            [0.7, 0.2, 0.1, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.09, 0.01],
            [0.6, 0.2, 0.1, 0.09, 0.01],
            [0.09, 0.5, 0.2, 0.2, 0.01],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ])
        output = dist_loss(x, target)

        ok = True
        for i in range(len(output)-1):
            if output[i] > output[i+1]:
                ok = False

            self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main(verbosity=2)
