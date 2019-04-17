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

    def __init__(self, shape, num_layers, span, device):
        '''
        shape: how many nodes in each direction to define a len(shape) dimensional grid of input wires.
            Some of these may just be used as placeholders/memory for intermediate computations
        nlayers: how many layers of processing before returning the final results
        span: How many nodes in each dimension to span
        '''
        super(ASIC, self).__init__()
        self.device = device
        dimension = len(shape)
        self.kernel = tuple(s * 2 + 1 for s in span)
        self.shape = shape
        self.layers = num_layers
        ninputs = numpy.prod(self.kernel)
        n_possible_inputs = 2 ** ninputs
        self.toggle_gates = torch.nn.Parameter(torch.rand(*(num_layers, n_possible_inputs) + shape, device=self.device, dtype=torch.float))
        self.bitmask = torch.from_numpy(numpy.asarray(list(itertools.product(range(2), repeat=ninputs)))).float().to(self.device).transpose(0, 1)
        self.bitmask = repeat(self.bitmask, shape)

    def convolve(self, x):
        '''
        slide and wrap self.kernel across x
        self.kernel.shape = size1, size2, ..., sizeN
        x.shape = batch_size, dimension1, dimension2, ..., dimensionN
        convolve(x) = batch_size,dimension1, dimension2, ..., dimensionN, size1 x size2 x ... x sizeN
        '''
        shape = x.shape
        for dimension, k in enumerate(self.kernel):
            if dimension:
                x = x.transpose(1, dimension + 1)
            inputs = []
            for i in range(k):
                inputs.append(torch.cat((x[:, i:], x[:, :i]), 1))
            x = torch.stack(inputs)
            x = first_to_last(x)
            if dimension:
                x = x.transpose(1, dimension + 1)
        x = x.reshape(shape + (-1,))
        # recenter
        center = -numpy.prod(tuple((k - 1) // 2 for k in self.kernel))
        ndim = len(x.shape)
        x = x.permute((ndim - 2, ndim - 1) + tuple(range(ndim - 2)))
        x = torch.cat((x[center:], x[:center]), 0)
        x = x.permute(tuple(range(2, ndim)) + (0, 1))
        return x

    def forward(self, x):
        '''
        forward pass through asic
        Bits on each wire are floating points between 0 and 1
        The output of each gate is a weighted average of the outputs of the gate over all possible inputs 
        the weights are determined by how close the actual floating point input is to each possible combination of boolean inputs
        The weights are derived from 1 - |bitmask - real_inputs|, where the bitmask contains an array of all possible combinations of inputs
        '''
        toggle_weights = self.toggle_gates.sigmoid()
        bitmask = repeat(self.bitmask, (x.shape[0],))
        slices = self.embed(x)
        for layer in range(self.layers):
            convolved = self.convolve(self.state)
            weight = (1 - torch.abs(last_to_first(bitmask) - convolved)).prod(-1).transpose(0, 1)
            self.state = (weight * toggle_weights[layer]).sum(1)
            self.state = torch.clamp(self.state, 0, 1)
        return self.state[slices]

    def apply(self, x):
        '''
        Similar to forward, except round the outputs at each layer
        This represents the real asic derived from the differentiable floating point version that is used for training
        '''
        toggle_weights = self.toggle_gates.sigmoid()
        bitmask = repeat(self.bitmask, (x.shape[0],))
        slices = self.embed(x)
        circuit = self.state.round()
        for i, layer in enumerate(range(self.layers)):
            convolved = self.convolve(circuit)
            weight = (1 - torch.abs(last_to_first(bitmask) - convolved)).prod(-1).transpose(0, 1)
            circuit = (weight * toggle_weights[layer]).sum(1)
            circuit = torch.clamp(circuit, 0, 1)
            circuit = circuit.round()
        return circuit[slices]

    def embed(self, x):
        '''
        evenly embed inputs into a multidimensional grid of inputs with shape self.shape
        inputs that are not assigned an element of x are used for temporary storage/memory
        Each element of self.shape must be a multiple of the corresponding elemento of x.shape
        '''
        self.state = torch.zeros((x.shape[0],) + self.shape, device=self.device)
        slices = [slice(None, None, None)]
        for my_shape, your_shape in zip(self.shape, x.shape[1:]):
            assert not my_shape % your_shape
            slices.append(slice(None, None, my_shape // your_shape))
        slices = tuple(slices)
        self.state[slices] = x
        return slices

bce = torch.nn.BCELoss()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = ASIC((16,), 5, (2,), device)

def f(x):
    ret = abs((x[:, 1] * x[:, 0]).unsqueeze(-1) - x)
    return ret.float()

epochs = 10 ** 6
optimizer = torch.optim.Adam(model.parameters())
batch_size = 256
memory = 2 # for 1 / 2 wires used for memory
model = model.to(device)
for epoch in range(epochs):
    optimizer.zero_grad()
    x = torch.randint(0, 2, size=(batch_size,) + tuple(s // memory for s in model.shape), device=device, dtype=torch.float)
    pred = model(x)
    pred_circuit = model.apply(x)
    true = f(x)
    loss = bce(pred, true)
    loss.backward()
    optimizer.step()
    if not epoch % 100:
        inputs = x[0]
        circuit_prediction = pred_circuit[0]
        true_output = true[0]
        accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
        this_loss = loss.item()
        inputs = inputs.detach().cpu().numpy()
        circuit_prediction = circuit_prediction.detach().cpu().numpy()
        true_output = true_output.detach().cpu().numpy()
        print('inpt:', inputs)
        print('pred:', circuit_prediction)
        print('true:', true_output)
        print('%accuracy:', accuracy)
        print('loss:', loss.item())
