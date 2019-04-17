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
        for layer in range(self.layers):
            convolved = self.convolve(self.state)
            weight = (1 - torch.abs(last_to_first(bitmask) - convolved)).prod(-1).transpose(0, 1)
            self.state = (weight * toggle_weights[layer]).sum(1)
            self.state = torch.clamp(self.state, 0, 1)
        return self.state[slices]

    def apply(self, x):
        '''
        apply asic circuit
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
model = ASIC((16,), 8, (2,), device)

def f(x):
    ret = abs((x[:, 1] * x[:, 0]).unsqueeze(-1) - x)
    return ret.float()

epochs = 10 ** 6
optimizer = torch.optim.Adam(model.parameters())
batch_size = 128
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
