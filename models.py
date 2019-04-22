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

    def __init__(self,
            shape,
            num_layers,
            kernel,
            device,
            recurrent=True,
            weight_sharing=False,
            kernel_offset='center'):
        '''
        shape: how many nodes in each direction to define a len(shape) dimensional grid of input wires.
            Some of these may just be used as placeholders/memory for intermediate computations
        num_layers: how many layers of processing before returning the final results
        kernel: How many nodes wide window should be in each dimension
        recurrent: Reset state before each forward pass or application
        weight_sharing: Whether to share free parameters across convolutions
        kernel_offset: whether a given input bit should be centered:
        - in the center of its convolutional window
        - to the left of its convolutional window
        - to the right of its convolutional window
        '''
        super(ASIC, self).__init__()
        self.device = device
        self.weight_sharing = weight_sharing
        self.recurrent = recurrent
        dimension = len(shape)
        if isinstance(kernel_offset, str):
            kernel_offset = (kernel_offset,) * dimension
        self.kernel_offset = kernel_offset
        if isinstance(kernel, int):
            kernel = (kernel,) * dimension
        self.kernel = kernel
        self.shape = shape
        self.layers = num_layers
        ninputs = numpy.prod(self.kernel)
        n_possible_inputs = 2 ** ninputs
        if self.weight_sharing:
            self.toggle_gates = torch.nn.Parameter(torch.rand(*(num_layers, n_possible_inputs), device=self.device, dtype=torch.float))
        else:
            self.toggle_gates = torch.nn.Parameter(torch.rand(*(num_layers, n_possible_inputs) + shape, device=self.device, dtype=torch.float))
        self.bitmask = torch.from_numpy(numpy.asarray(list(itertools.product(range(2), repeat=ninputs)))).float().to(self.device).transpose(0, 1)
        self.bitmask = repeat(self.bitmask, shape)

    def convolve(self, x):
        '''
        slide and wrap self.kernel across x
        self.kernel = size1, size2, ..., sizeN
        x.shape = batch_size, dimension1, dimension2, ..., dimensionN
        convolve(x) = batch_size,dimension1, dimension2, ..., dimensionN, size1 x size2 x ... x sizeN
        '''
        shape = x.shape
        for dimension, (k, offset) in enumerate(zip(self.kernel, self.kernel_offset)):
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
        if offset == 'center':
            center = -numpy.prod(tuple((k - 1) // 2 for k in self.kernel))
        elif offset == 'left':
            center = 0
        elif offset == 'right':
            center = -numpy.prod(tuple((k - 1) for k in self.kernel))
        else:
            raise Exception('Invalid kernel_offset')
        ndim = len(x.shape)
        x = x.permute((ndim - 2, ndim - 1) + tuple(range(ndim - 2)))
        x = torch.cat((x[center:], x[:center]), 0)
        x = x.permute(tuple(range(2, ndim)) + (0, 1))
        return x

    def forward(self, x, state=None, harden=False):
        '''
        forward pass through asic
        Bits on each wire are floating points between 0 and 1
        The output of each gate is a weighted average of the outputs of the gate over all possible inputs 
        the weights are determined by how close the actual floating point input is to each possible combination of boolean inputs
        The weights are derived from 1 - |bitmask - real_inputs|, where the bitmask contains an array of all possible combinations of inputs
        '''
        toggle_weights = self.toggle_gates.sigmoid()
        if self.weight_sharing:
            toggle_weights = repeat(toggle_weights, self.shape)
            for _ in enumerate(self.shape):
                toggle_weights = first_to_last(toggle_weights)
        bitmask = last_to_first(repeat(self.bitmask, (x.shape[0],)))
        if state is None:
            state = self.state
        state, slices = self.embed(x, state)
        hardened_output = self.state.round()
        for layer in range(self.layers):
            convolved = self.convolve(self.state)
            weight = (1 - torch.abs(bitmask - convolved)).prod(-1).transpose(0, 1)
            self.state = (weight * toggle_weights[layer]).sum(1)
            self.state = torch.clamp(self.state, 0, 1)
            if harden:
                self.state = self.state.round()
        return self.state[slices]

    def apply(self, x):
        '''
        Similar to forward, except round the outputs at each layer
        This represents the real asic derived from the differentiable floating point version that is used for training
        '''
        return self.forward(x, harden=True)

    def embed(self, state, x):
        '''
        evenly embed x into a multidimensional grid of inputs with shape self.shape
        inputs that are not assigned an element of x are used for temporary storage/memory
        Each element of self.shape must be a multiple of the corresponding elemento of x.shape
        '''
        if hasattr(self, 'state'):
            if not self.recurrent:
                state = self.state.detach()
                print('detached')
            # batch_size changed
            if len(self.state) < len(x):
                zeros = torch.zeros((len(x) - len(self.state),) + self.shape, device=self.device)
                state = torch.cat((self.state, zeros), 0)
            elif len(self.state) > len(x):
                state = self.state[:len(x)]
        else:
            state = torch.zeros((len(x),) + self.shape, device=self.device)
        slices = [slice(None, None, None)]
        for my_shape, your_shape in zip(self.shape, x.shape[1:]):
            memory = my_shape - your_shape
            if not my_shape % your_shape:
                slices.append(slice(None, None, my_shape // your_shape))
            elif not my_shape % memory:
                indices = torch.ones(my_shape, dtype=torch.uint8)
                indices[slice(None, None, my_shape // memory)] = 0
                slices.append(indices)
            else:
                raise
        slices = tuple(slices)
        state[slices] = x
        return state, slices

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
