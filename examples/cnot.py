import torch

from models import ASIC, device
from train import stochastic

def target(x):
    ret = abs((x[:, 1] * x[:, 0]).unsqueeze(-1) - x)
    return ret.float()

model = ASIC((16,), 5, (2,), device)
batch_size = 64
stochastic(model, target, (batch_size, 8), 10 ** 6)
