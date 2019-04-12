            print('shae')
            print(new_outputs[i].shape)
import itertools
import numpy
import torch

def score_max(x, dim, score):
    _tmp = [1] * len(x.size())
    _tmp[dim] = x.size(dim)
    return torch.gather(x, dim, score.unsqueeze(dim).repeat(tuple(_tmp))).select(dim, 0)

class ASIC(torch.nn.Module):

    def __init__(self, shape, kernel):
        # dimensions: horizontal, vertical x direction -1, direction +1 x incoming bits from other rails x n x n
        super(ASIC, self).__init__()
        self.dimension = len(shape)
        self.kernel = kernel
        self.toggle_gates = torch.nn.Parameter(torch.rand(*numpy.prod(kernel) + shape))

    def forward(self, x):
        '''
        takes input and output
        '''
        outputs = x
        for  in self.
            self.rail_state.reshape(self.rail_shape)[1, 1, :len(x), 0] = x
        new_outputs = torch.empty((numpy.prod(self.mask_shape[:2]),) + self.ns)
        new_inputs = self.rail_state[self.input_mask].reshape((-1,) + self.ns)
        toggle_weights = torch.nn.Sigmoid()(self.toggle_gates)
        for i, inputs in enumerate(new_inputs):
            inputs_i = torch.cat((new_inputs[:i], new_inputs[i + 1:]), 0)
            weight = (1 - torch.abs(self.bitmask - inputs_i)).prod(1)
            # toggled = toggle_weights[i] * inputs + (1 - toggle_weights[i]) * (1 - inputs)
            toggled = toggle_weights[i]
            # new_outputs[i] = (toggled * weight).sum(0)
            new_outputs[i] = score_max(toggled, 0, weight.argmax(0))
        new_outputs = torch.clamp(new_outputs.reshape(-1), 0, 1)

            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
            print('shae')
            print(new_outputs[i].shape)
        self.counter += 1
        self.counter %= self.recurrent_count
        self.rail_state[self.output_mask] = new_outputs

        return new_outputs[mask]

model = ASIC(3)
mask = torch.zeros(model.output_mask.sum(), dtype=torch.uint8)
mask[-3:] = 1

def f(x):
    ret = abs(x[1] * x[0] - x)
    return ret.float()
epochs = 10000
optimizer = torch.optim.Adam(model.parameters())
batch_size = 32
tally = 1
for _ in range(epochs):
    loss = 0
    optimizer.zero_grad()
    for _ in range(batch_size):
        # x = torch.from_numpy(numpy.asarray([0, 1, 0]))
        x = torch.from_numpy(numpy.random.randint(0, 2, size=3))
        model.reset_rails()
        v = model(x, mask)
        for _ in range(model.ns[-1]):
            v = model([], mask)
        u = f(x)
        tally += u
        loss = ((abs(v - u)) * 1. / tally).sum() / (1. / tally).sum()
        # loss = abs(v - u).mean()
        loss.backward(retain_graph=True)
    print(x)
    print(v)
    print(f(x))
    print(loss.item() / batch_size)
    optimizer.step()
