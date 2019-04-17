import itertools
import numpy
import torch

def score_max(x, dim, score):
    _tmp = [1] * len(x.size())
    _tmp[dim] = x.size(dim)
    return torch.gather(x, dim, score.unsqueeze(dim).repeat(tuple(_tmp))).select(dim, 0)

def repeat(x, shape):
    return x.repeat(shape + (1,) * len(x.shape))

def last_to_first(x):
    ndim = len(x.shape)
    return x.permute((ndim - 1,) + tuple(range(ndim - 1)))

def first_to_last(x):
    ndim = len(x.shape)
    return x.permute(tuple(range(1, ndim)) + (0,))

class ASIC(torch.nn.Module):

    def __init__(self, shape, layers, kernel):
        # dimensions: horizontal, vertical x direction -1, direction +1 x incoming bits from other rails x n x n
        super(ASIC, self).__init__()
        dimension = len(shape)
        self.kernel = kernel
        self.shape = shape
        self.layers = layers
        ninputs = numpy.prod(kernel)
        n_possible_inputs = 2 ** ninputs
        self.toggle_gates = torch.nn.Parameter(torch.rand(*(layers, n_possible_inputs) + shape))
        self.bitmask = torch.from_numpy(numpy.asarray(list(itertools.product(range(2), repeat=ninputs)))).transpose(0, 1)
        self.bitmask = repeat(self.bitmask, shape).float()

    def convolve(self, x):
        shape = x.shape
        for dimension, k in enumerate(self.kernel):
            if dimension:
                x = x.transpose(1, dimension + 1)
            inputs = []
            for i in range(k):
                inputs.append(torch.cat((x[:, i:], x[: ,:i]), 1))
            x = torch.stack(inputs)
            x = x.permute(tuple(range(1, len(x.shape))) + (0,))
            if dimension:
                x = x.transpose(1, dimension + 1)
        return x.reshape(shape + (-1,))

    def forward(self, x):
        '''
        forward pass through asic
        '''
        toggle_weights = self.toggle_gates.sigmoid()
        bitmask = repeat(self.bitmask, (x.shape[0],))
        slices = self.embed(x)
        for i, layer in enumerate(range(self.layers)):
            convolved = self.convolve(self.state)
            weight = (1 - torch.abs(last_to_first(bitmask) - convolved)).prod(-1).transpose(0, 1)
            self.state = (weight * toggle_weights[layer]).sum(1)
            self.state = torch.clamp(self.state, 0, 1)
        return self.state[slices]

    def embed(self, x):
        self.state = torch.zeros((x.shape[0],) + self.shape)
        slices = [slice(None, None, None)]
        for my_shape, your_shape in zip(self.shape, x.shape[1:]):
            assert not my_shape % your_shape
            slices.append(slice(None, None, my_shape // your_shape))
        slices = tuple(slices)
        self.state[slices] = x
        return slices

bce = torch.nn.BCELoss()

model = ASIC((8,), 8, (3,))

def f(x):
    ret = abs((x[:, 1] * x[:, 0]).unsqueeze(-1) - x)
    return ret.float()

epochs = 100000
optimizer = torch.optim.Adam(model.parameters())
batch_size = 128
memory = 2
for epoch in range(epochs):
    optimizer.zero_grad()
    # x = torch.from_numpy(numpy.asarray([0, 1, 0]))
    x = torch.from_numpy(numpy.random.randint(0, 2, size=(batch_size,) + tuple(s // memory for s in model.shape)))
    pred = model(x.float())
    true = f(x)
    loss = bce(pred, true) #+ regularizer
    loss.backward()
    if not epoch % 100:
        print(x[0])
        print(pred[0])
        print(pred[0].round())
        print(true[0])
        print(1 - abs(true - pred.round()).mean().item())
        print(loss.item())
    optimizer.step()
