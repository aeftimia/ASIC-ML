import itertools
import numpy
import torch

from models import ASIC

def test_convolve():
    x = torch.as_tensor([[0,1,2,3,4,5,6,7]])
    model = ASIC((8,), 1, (5,), 'cpu')
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.as_tensor([[
        [6,7,0,1,2],
        [7,0,1,2,3],
        [0,1,2,3,4],
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [4,5,6,7,0],
        [5,6,7,0,1]]])))

    model = ASIC((8,), 1, (5,), 'cpu', kernel_offset='left')
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.as_tensor([[
        [0,1,2,3,4],
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [4,5,6,7,0],
        [5,6,7,0,1],
        [6,7,0,1,2],
        [7,0,1,2,3]
        ]])))

    model = ASIC((8,), 1, (5,), 'cpu', kernel_offset='right')
    convolved = model.convolve(x)
    assert torch.all(torch.eq(convolved, torch.as_tensor([[
        [4,5,6,7,0],
        [5,6,7,0,1],
        [6,7,0,1,2],
        [7,0,1,2,3],
        [0,1,2,3,4],
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7]
        ]])))


    model = ASIC((2, 8), 1, (2, 5), 'cpu', kernel_offset='right')
    convolved = model.convolve(torch.stack((x, x + 8), 1))
    print('convolved')
    print(convolved.shape)
    print(convolved)
    assert torch.all(torch.eq(convolved, torch.as_tensor(
        [[[[12, 13, 14, 15,  8,  4,  5,  6,  7,  0],
            [13, 14, 15,  8,  9,  5,  6,  7,  0,  1],
            [14, 15,  8,  9, 10,  6,  7,  0,  1,  2],
            [15,  8,  9, 10, 11,  7,  0,  1,  2,  3],
            [ 8,  9, 10, 11, 12,  0,  1,  2,  3,  4],
            [ 9, 10, 11, 12, 13,  1,  2,  3,  4,  5],
            [10, 11, 12, 13, 14,  2,  3,  4,  5,  6],
            [11, 12, 13, 14, 15,  3,  4,  5,  6,  7]],

            [[ 4,  5,  6,  7,  0, 12, 13, 14, 15,  8],
                [ 5,  6,  7,  0,  1, 13, 14, 15,  8,  9],
                [ 6,  7,  0,  1,  2, 14, 15,  8,  9, 10],
                [ 7,  0,  1,  2,  3, 15,  8,  9, 10, 11],
                [ 0,  1,  2,  3,  4,  8,  9, 10, 11, 12],
                [ 1,  2,  3,  4,  5,  9, 10, 11, 12, 13],
                [ 2,  3,  4,  5,  6, 10, 11, 12, 13, 14],
                [ 3,  4,  5,  6,  7, 11, 12, 13, 14, 15]]]])))

    model = ASIC((2, 8), 1, (2, 5), 'cpu', kernel_offset=('left', 'right'))
    convolved = model.convolve(torch.stack((x, x + 8), 1))
    print('convolved')
    print(convolved.shape)
    print(convolved)
    assert torch.all(torch.eq(convolved, torch.as_tensor(
        [[[[ 4,  5,  6,  7,  0, 12, 13, 14, 15,  8],
            [ 5,  6,  7,  0,  1, 13, 14, 15,  8,  9],
            [ 6,  7,  0,  1,  2, 14, 15,  8,  9, 10],
            [ 7,  0,  1,  2,  3, 15,  8,  9, 10, 11],
            [ 0,  1,  2,  3,  4,  8,  9, 10, 11, 12],
            [ 1,  2,  3,  4,  5,  9, 10, 11, 12, 13],
            [ 2,  3,  4,  5,  6, 10, 11, 12, 13, 14],
            [ 3,  4,  5,  6,  7, 11, 12, 13, 14, 15]],

            [[12, 13, 14, 15,  8,  4,  5,  6,  7,  0],
                [13, 14, 15,  8,  9,  5,  6,  7,  0,  1],
                [14, 15,  8,  9, 10,  6,  7,  0,  1,  2],
                [15,  8,  9, 10, 11,  7,  0,  1,  2,  3],
                [ 8,  9, 10, 11, 12,  0,  1,  2,  3,  4],
                [ 9, 10, 11, 12, 13,  1,  2,  3,  4,  5],
                [10, 11, 12, 13, 14,  2,  3,  4,  5,  6],
                [11, 12, 13, 14, 15,  3,  4,  5,  6,  7]]]]
            )))

def test_embed():
    x = torch.as_tensor([[2,] * 8]).float()
    model = ASIC((8,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state == 2)
    model = ASIC((16,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, ::2] == 2)
    model = ASIC((12,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.as_tensor([0, 1, 1] * 4, dtype=torch.uint8)] == 2)
    model = ASIC((10,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.as_tensor([0, 1, 1, 1, 1] * 2, dtype=torch.uint8)] == 2)
    model = ASIC((9,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.as_tensor([0] + [1] * 8, dtype=torch.uint8)] == 2)
    assert torch.all(state[:, 0] == 0)
