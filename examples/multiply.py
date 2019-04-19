import numpy
import torch

from models import ASIC, device
from train import stochastic

def target(x):
    basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
    numbers = torch.tensordot(x, basis, ([2], [0]))
    product = numbers.prod(-1)
    ret = torch.zeros((x.shape[0], numpy.prod(x.shape[1:])), device=x.device)
    for i in range(numpy.prod(x.shape[1:])):
        ret[(product % 2) == 1, i] = 1
        product //= 2
    return ret.reshape(x.shape).float()

n_integers = 2
m_bits_per_integer = 16
model = ASIC((n_integers, m_bits_per_integer * 3 // 2,), 2, (3, 3), device, kernel_offset='right')
batch_size = 32
stochastic(model,
        target,
        (batch_size,
            n_integers,
            m_bits_per_integer),
        10 ** 6)
