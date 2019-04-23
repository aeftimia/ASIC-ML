import torch

from models import ASIC, device
from train import stochastic

def target(x):
    basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
    num0 = torch.mv(x[:, 0], basis)
    num1 = torch.mv(x[:, 1], basis)
    num = (num0 + num1) % basis.sum()
    ret = torch.zeros((x.shape[0], 1, x.shape[2]))
    for i, _ in enumerate(basis):
        ret[(num % 2) == 1, 0, i] = 1
        num //= 2
    ret = torch.cat((ret, torch.zeros((ret.shape[0], x.shape[1] - ret.shape[1]) + ret.shape[2:])), dim=1)
    return ret.float()

def sample(shape):
    x = torch.randint(0,
            2,
            size=shape,
            device=model.device,
            dtype=torch.float)
    x[:, 2:] = 0
    return x

model = ASIC((3, 8),
        1,
        (3, 2),
        device,
        kernel_offset='right', weight_sharing=(False, True))

batch_size = 8
shape = (batch_size,) + model.shape
bce = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
rolling_loss = 0
rolling_accuracy = 0
max_passes = 6
epochs = 10 ** 6
checkpoint = 100
for epoch in range(epochs):
    x = sample(shape)
    inpt = x
    pred_circuit = x.round()
    true = target(inpt)
    optimizer.zero_grad()
    for i in range(max_passes):
        pred = model(x)
        pred_circuit = model(pred_circuit, hard=True)
        x = pred
    loss = bce(pred, true)
    loss.backward()
    optimizer.step()
    rolling_loss *= (1 - 1. / checkpoint)
    rolling_loss +=  1. / checkpoint * loss.item()
    accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
    rolling_accuracy *= (1 - 1. / checkpoint)
    rolling_accuracy +=  1. / checkpoint * accuracy
    if not epoch % checkpoint:
        print('{0}/{1} passes'.format(i + 1, max_passes))
        inputs = inpt[0]
        circuit_prediction = pred_circuit[0]
        true_output = true[0]
        this_loss = loss.item()
        inputs = inputs.detach().cpu().numpy()
        circuit_prediction = circuit_prediction.detach().cpu().numpy()
        true_output = true_output.detach().cpu().numpy()
        print('epoch:', epoch)
        print('inpt:', inputs)
        print('pred:', circuit_prediction)
        print('true:', true_output)
        print('%accuracy:', rolling_accuracy)
        print('loss:', rolling_loss)
