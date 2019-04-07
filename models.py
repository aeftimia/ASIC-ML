import itertools
import numpy
import torch

class ASIC(torch.nn.Module):

    def __init__(self, n):
        # dimensions: horizontal, vertical x direction -1, direction +1 x incoming bits from other rails x n x n
        super(ASIC, self).__init__()
        dimension = 2
        ns = (n,) * dimension
        self.toggle_gates = torch.nn.Parameter(torch.rand(*(2 * dimension, 2 ** (2 * dimension - 1),) + ns))
        self.rail_shape = (2, dimension) + (n + 1,) * dimension
        self.mask_shape = (2, dimension) + (n,) * dimension
        self.rail_state = torch.rand(numpy.prod(self.rail_shape))
        self.bitmask = torch.from_numpy(numpy.asarray(list(itertools.product(range(2), repeat=2 * dimension - 1))))
        self.bitmask = self.bitmask.repeat(ns + (1, 1)).permute((dimension, dimension + 1) + tuple(range(dimension))).float()

        self.input_mask = numpy.zeros(self.rail_shape, dtype='uint8')
        self.input_mask[(slice(0, 1, 1), slice(None, None, 1)) + tuple(slice(1, None, 1) for _ in range(dimension))] = 1
        self.input_mask[(slice(1, 2, 1), slice(None, None, 1)) + tuple(slice(0, -1, 1) for _ in range(dimension))] = 1
        self.input_mask = torch.from_numpy(self.input_mask.reshape(-1))

        self.output_mask = numpy.zeros(self.rail_shape, dtype='uint8')
        self.output_mask[(slice(0, 1, 1), slice(None, None, 1)) + tuple(slice(0, -1, 1) for _ in range(dimension))] = 1
        self.output_mask[(slice(1, 2, 1), slice(None, None, 1)) + tuple(slice(1, None, 1) for _ in range(dimension))] = 1
        self.output_mask = torch.from_numpy(self.output_mask.reshape(-1))
        self.ns = ns

        diagonal = torch.zeros(self.rail_state.shape[0])
        diagonal[self.output_mask.nonzero()] = 1
        self.assignment_matrix = torch.diag(diagonal, 0)[:, :self.output_mask.sum()]

    def forward(self, x, mask):
        '''
        takes input and output
        '''
        self.rail_state.reshape(self.rail_shape)[1, 1, :len(x), 0] = x
        new_outputs = torch.empty((numpy.prod(self.mask_shape[:2]),) + self.ns)
        new_inputs = self.rail_state[self.input_mask].reshape((-1,) + self.ns)
        toggle_weights = torch.nn.Sigmoid()(self.toggle_gates)
        for i, inputs in enumerate(new_inputs):
            inputs_i = torch.cat((new_inputs[:i], new_inputs[i + 1:]), 0)
            weight = (1 - torch.abs(self.bitmask - inputs_i)).prod(1)
            toggled = toggle_weights[i] * inputs + (1 - toggle_weights[i]) * (1 - inputs)
            new_outputs[i] = (toggled * weight).sum(0)
        new_outputs = torch.clamp(new_outputs.reshape(-1), 0, 1)
        self.rail_state[self.output_mask] = new_outputs
        return self.rail_state[mask]*self.toggle_gates.reshape(-1)[0]

model = ASIC(10)
num_rails = len(model.rail_state)
mask = torch.from_numpy(numpy.zeros(model.rail_state.shape, dtype='uint8'))
mask[num_rails - 3:num_rails] = 1

def f(x):
    ret = x ** 2
    return ret.float()
epochs = 1000
optimizer = torch.optim.Adam(model.parameters())
for _ in range(epochs):
    x = torch.from_numpy(numpy.asarray([0, 1, 0]))
    v = model(x, mask)
    print(v)
    print(f(x))
    optimizer.zero_grad()
    loss = ((v - f(x)) ** 2).mean()
    print(loss.item())
    loss.backward()
    optimizer.step()
