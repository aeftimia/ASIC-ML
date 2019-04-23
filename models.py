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
            recure=1,
            weight_sharing=False,
            kernel_offset='center'):
        '''
        shape: how many nodes in each direction to define a len(shape) dimensional grid of input wires.
            Some of these may just be used as placeholders/memory for intermediate computations
        num_layers: how many layers of processing before returning the final results
        kernel: How many nodes wide window should be in each dimension
        weight_sharing: Whether to share free parameters across convolutions
        kernel_offset: whether a given input bit should be centered:
        - in the center of its convolutional window
        - to the left of its convolutional window
        - to the right of its convolutional window
        recure: How many times to feed the output of the model back in
        '''
        super(ASIC, self).__init__()
        self.device = device
        self.recure = recure
        dimension = len(shape)
        if not isinstance(weight_sharing, tuple):
            weight_sharing = (weight_sharing,) * dimension
        self.weight_sharing = weight_sharing
        if not isinstance(kernel_offset, tuple):
            kernel_offset = (kernel_offset,) * dimension
        self.kernel_offset = kernel_offset
        if not isinstance(kernel, tuple):
            kernel = (kernel,) * dimension
        self.kernel = kernel
        self.shape = shape
        self.layers = num_layers
        ninputs = numpy.prod(self.kernel)
        n_possible_inputs = 2 ** ninputs
        unshared = []
        for weight_sharing, dim in zip(self.weight_sharing, self.shape):
            if not weight_sharing:
                unshared.append(dim)
            self.toggle_gates = torch.nn.Parameter(torch.rand(*(num_layers, n_possible_inputs) + tuple(unshared), device=self.device, dtype=torch.float))
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
        center = -1
        for dimension, (k, offset) in enumerate(zip(self.kernel, self.kernel_offset)):
            # recenter
            if offset == 'center':
                center = (k - 1) // 2
            elif offset == 'left':
                center = 0
            elif offset == 'right':
                center = k - 1
            else:
                raise Exception('Invalid kernel_offset')
            if dimension:
                x = x.transpose(1, dimension + 1)
            inputs = []
            for i in range(k):
                feed = torch.cat((x[:, i:], x[:, :i]), 1)
                feed = torch.cat((feed[:, -center:], feed[:, :-center]), 1)
                inputs.append(feed)
            x = torch.stack(inputs)
            x = first_to_last(x)
            if dimension:
                x = x.transpose(1, dimension + 1)
        x = x.reshape(shape + (-1,))
        ndim = len(x.shape)
        x = x.permute((ndim - 2, ndim - 1) + tuple(range(ndim - 2)))
        x = x.permute(tuple(range(2, ndim)) + (0, 1))
        return x

    def get_toggle_weights(self):
        toggle_weights = self.toggle_gates.sigmoid()
        for weight_sharing, dim in zip(self.weight_sharing, self.shape):
            if weight_sharing:
                toggle_weights = first_to_last(repeat(toggle_weights, (dim,)))
        return toggle_weights

    def forward(self, x, harden=False):
        '''
        forward pass through asic
        Bits on each wire are floating points between 0 and 1
        The output of each gate is a weighted average of the outputs of the gate over all possible inputs 
        the weights are determined by how close the actual floating point input is to each possible combination of boolean inputs
        The weights are derived from 1 - |bitmask - real_inputs|, where the bitmask contains an array of all possible combinations of inputs
        '''
        toggle_weights = self.get_toggle_weights()
        bitmask = last_to_first(repeat(self.bitmask, (x.shape[0],)))
        state, mask = self.embed(x)
        for _ in range(self.recure):
            for layer in range(self.layers):
                convolved = self.convolve(state)
                weight = (1 - torch.abs(bitmask - convolved)).prod(-1).transpose(0, 1)
                state = (weight * toggle_weights[layer]).sum(1)
                state = torch.clamp(state, 0, 1)
                if harden:
                    state = state.round()
        return state[mask].reshape(x.shape)

    def embed(self, x, state=None):
        '''
        evenly embed x into a multidimensional grid of inputs with shape self.shape
        inputs that are not assigned an element of x are used for temporary storage/memory
        Each element of self.shape must be a multiple of the corresponding elemento of x.shape
        '''
        if state is not None:
            # batch_size changed
            if len(state) < len(x):
                zeros = torch.zeros((len(x) - len(state),) + self.shape, device=self.device)
                state = torch.cat((state, zeros), 0)
            elif len(state) > len(x):
                state = state[:len(x)]
        else:
            state = torch.zeros((len(x),) + self.shape, device=self.device)
        mask = torch.ones(state.shape, dtype=torch.uint8, device=self.device)
        for dim, (my_shape, your_shape) in enumerate(zip(self.shape, x.shape[1:])):
            dim += 1
            mask = mask.transpose(0, dim)
            memory = my_shape - your_shape
            if not my_shape % your_shape:
                di = my_shape // your_shape
                for i in range(1, di):
                    mask[i::di] = 0
                mask = torch.clamp(mask, 0, 1)
            elif not my_shape % memory:
                di = my_shape // memory
                mask[::di] = 0
            else:
                raise Exception('Invalid embedding')
            mask = mask.transpose(0, dim)
        state.reshape(-1)[mask.reshape(-1)] = x.reshape(-1)
        return state, mask

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
