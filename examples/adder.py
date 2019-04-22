import torch

from models import ASIC, device
from train import stochastic

def target(x):
    basis = 2 ** torch.arange(x.shape[2], device=x.device).float()
    num0 = torch.mv(x[:, 0], basis)
    num1 = torch.mv(x[:, 1], basis)
    num = (num0 + num1) % (basis.sum() + 1)
    ret = torch.zeros((x.shape[0], x.shape[2]))
    for i, _ in enumerate(basis):
        ret[(num % 2) == 1, i] = 1
        num //= 2
    return ret.float()

model = ASIC((3, 3),
        1,
        (3, 3),
        device,
        kernel_offset='right',
        weight_sharing=True,
        recurrent=False)
batch_size = 128
shape = (batch_size, 3, 3)

bce = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
rolling_loss = 0
rolling_accuracy = 0
max_passes = 10
epochs = 10 ** 6
checkpoint = 1
for epoch in range(epochs):
    if hasattr(model, 'state'):
        del model.state
    x = torch.randint(0,
            2,
            size=shape,
            device=model.device,
            dtype=torch.float)
    x[..., 2] = 0
    inpt = x
    pred_circuit = x.round()
    true = target(inpt)
    for i in range(max_passes):
        print(i)
        optimizer.zero_grad()
        pred = model(x)
        res = pred[:, 0]
        stop = pred[:, 1:]
        loss_res = bce(res, true)
        acc_res = 1 - abs(res - true).mean()
        loss_stop = bce(stop, torch.zeros_like(stop))
        acc_stop = 1 - stop.mean()
        loss = acc_res * loss_stop + acc_stop * loss_res + 1 - (acc_res + acc_stop) / 2
        x = pred
        loss.backward(retain_graph=True)
        optimizer.step()
        pred_circuit = model.apply(pred_circuit)
        if stop.sum() < 0.5:
            break
    pred_circuit = pred_circuit[:, 0]
    rolling_loss *= (1 - 1. / checkpoint)
    rolling_loss +=  1. / checkpoint * loss.item()
    accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
    rolling_accuracy *= (1 - 1. / checkpoint)
    rolling_accuracy +=  1. / checkpoint * accuracy
    if not epoch % checkpoint:
        print('{0}/{1} passes'.format(i + 1, max_passes))
        inputs = inpt[0, :2]
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
