import itertools
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
    model = ASIC((9,), 1, (5,), 'cpu')
    state, slices = model.embed(x)
    assert torch.all(state[:, torch.from_numpy(numpy.asarray([1, 2, 3, 4, 5, 6, 7]))] == 2)
    assert torch.all(state[:, torch.from_numpy(numpy.asarray([0]))] == 0)

def test_adder():
    model = ASIC((2,4),
            1,
            (2,2),
            'cpu',
            kernel_offset='right',
            recure=4,
            weight_sharing=(False, True))
    # [[0,0],[0,0]], [[0,0],[0,1]], [[0,0],[1,0]], [[0,0],[1,1]],
    # [[0,1],[0,0]], [[0,1],[0,1]], [[0,1],[1,1]], [[1,0],[0,0]],
    # [[1,0],[0,0]], [[1,0],[0,1]], [[1,0],[1,0]], [[1,0],[1,1]],
    # [[1,1],[0,0]], [[1,1],[0,1]], [[1,1],[1,0]], [[1,1],[1,1]],
    # toggle_gates is 1x16x2
    # in last dimension, first entry is lower right, second is upper right
    model.toggle_gates.data = torch.from_numpy(numpy.asarray([[
        [0, 0], [0, 1], [0, 0], [0, 1],
        [0, 1], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0, 1], [0, 0], [1, 1], [0, 0]
        ]])).float()

    def target(x):
        basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
        num0 = torch.mv(x[:, 0], basis)
        num1 = torch.mv(x[:, 1], basis)
        num = (num0 + num1) % basis.sum()
        ret = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        for i, _ in enumerate(basis):
            ret[(num % 2) == 1, 0, i] = 1
            num //= 2
        ret = torch.cat(
                (ret,
                    torch.zeros(
                        (ret.shape[0], x.shape[1] - ret.shape[1]) + ret.shape[2:],
                    device=x.device)),
                dim=1)
        return ret.float()

    for x1, x2 in itertools.product(itertools.product(range(2), repeat=4), repeat=2):
        x1 = torch.from_numpy(numpy.asarray([x1]))
        x2 = torch.from_numpy(numpy.asarray([x2]))
        x = torch.stack((x1, x2), 1).float()
        y = target(x)
        pred = model(x, harden=True)
        print('inpt:', x.numpy())
        print('pred:', pred.detach().numpy())
        print('targ:', y.numpy())
        assert torch.all(pred == y)
