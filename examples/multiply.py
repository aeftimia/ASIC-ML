import numpy
import torch

from models import ASIC, device
from train import stochastic

def target(x):
    basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
    num0 = torch.mv(x[:, 0], basis)
    num1 = torch.mv(x[:, 1], basis)
    num = (num0 * num1) % (2 ** (x.shape[2] * 2))
    ret = torch.zeros((x.shape[0], x.shape[2] * 2,), device=x.device)
    for i in range(ret.shape[-1]):
        ret[(num % 2) == 1, i] = 1
        num //= 2
    return ret.float().reshape(x.shape)

model = ASIC((2, 12),
        3,
        (2, 5),
        device,
        kernel_offset='right',
        weight_sharing=(False, False),
        recure=1)

batch_size = 8
stochastic(model, target, (batch_size, 2, 8), 10 ** 6)
