import torch

from models import ASIC, device
from train import train

def target(x):
    basis = 2 ** torch.arange(x.shape[1], device=x.device).float()
    num = (torch.mv(x, basis) + 1) % (basis.sum() + 1)
    ret = torch.zeros_like(x)
    for i, _ in enumerate(basis):
        ret[(num % 2) == 1, i] = 1
        num //= 2
    return ret.float()

model = ASIC((16,), 8, (2,), device)
batch_size = 64
train(model, target, (batch_size, 8), 10 ** 6)
