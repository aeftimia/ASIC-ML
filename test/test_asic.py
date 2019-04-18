import numpy
import torch

from models import ASIC

def test_convolve():
    model = ASIC((8,), 1, (2,), 'cpu')
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

def test_embed():
    x = torch.from_numpy(numpy.asarray([[2,2,2,2,2,2,2,2]])).float()
    model = ASIC((8,), 1, (2,), 'cpu')
    slices = model.embed(x)
    assert torch.all(model.state == 2)
    model = ASIC((16,), 1, (2,), 'cpu')
    slices = model.embed(x)
    assert torch.all(model.state[:, ::2] == 2)
    model = ASIC((12,), 1, (2,), 'cpu')
    slices = model.embed(x)
    assert torch.all(model.state[:, torch.from_numpy(numpy.asarray([1, 2, 4, 5, 7, 8]))] == 2)
