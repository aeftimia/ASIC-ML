import numpy
import torch

from models import ASIC

def test_convolve():
    model = ASIC((8,), 1, (2,), 'cpu')
    batch_size = 1
    x = torch.from_numpy(numpy.asarray([[0,1,2,3,4,5,6,7]]))
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.from_numpy(
            numpy.asarray([[
                [6,7,0,1,2],
                [7,0,1,2,3],
                [0,1,2,3,4],
                [1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7],
                [4,5,6,7,0],
                [5,6,7,0,1]]]))))
