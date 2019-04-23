import numpy
import torch

from models import ASIC

def test_convolve():
    x = torch.from_numpy(numpy.asarray([[0,1,2,3,4,5,6,7]]))
    model = ASIC((8,), 1, (5,), 'cpu')
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

    model = ASIC((8,), 1, (5,), 'cpu', kernel_offset='left')
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.from_numpy(
            numpy.asarray([[
                [0,1,2,3,4],
                [1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7],
                [4,5,6,7,0],
                [5,6,7,0,1],
                [6,7,0,1,2],
                [7,0,1,2,3]
                ]]))))

    model = ASIC((8,), 1, (5,), 'cpu', kernel_offset='right')
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.from_numpy(
            numpy.asarray([[
                [4,5,6,7,0],
                [5,6,7,0,1],
                [6,7,0,1,2],
                [7,0,1,2,3],
                [0,1,2,3,4],
                [1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7]
                ]]))))

def test_embed():
    x = torch.from_numpy(numpy.asarray([[2,2,2,2,2,2,2,2]])).float()
    model = ASIC((8,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state == 2)
    model = ASIC((16,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, ::2] == 2)
    model = ASIC((12,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.from_numpy(numpy.asarray([1, 2, 4, 5, 7, 8, 10, 11]))] == 2)
    model = ASIC((10,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.from_numpy(numpy.asarray([1, 2, 3, 4, 6, 7, 8, 9]))] == 2)
