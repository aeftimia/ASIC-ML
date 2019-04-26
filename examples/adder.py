import torch

from models import ASIC, device
from train import stochastic

def target(x):
    basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
    num0 = torch.mv(x[:, 0], basis)
    num1 = torch.mv(x[:, 1], basis)
    num = (num0 + num1) % basis.sum()
    ret = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
    for i, _ in enumerate(basis):
        ret[(num % 2) == 1, 0, i] = 1
        num //= 2
    ret = torch.cat(
            (ret,
                torch.zeros(
                    (ret.shape[0], x.shape[1] - ret.shape[1]) + ret.shape[2:],
                device=x.device)),
            dim=1)
    return ret.float()

model = ASIC((2, 12),
        2,
        (2, 5),
        device,
        kernel_offset='right',
        weight_sharing=(False, False),
        recure=1)

batch_size = 8
stochastic(model, target, (batch_size, 2, 8), 10 ** 6)
