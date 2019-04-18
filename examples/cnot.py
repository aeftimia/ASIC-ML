import torch

from models import ASIC, device

def target(x):
    ret = abs((x[:, 1] * x[:, 0]).unsqueeze(-1) - x)
    return ret.float()

model = ASIC((16,), 5, (2,), device)
bce = torch.nn.BCELoss()

epochs = 10 ** 6
optimizer = torch.optim.Adam(model.parameters())
batch_size = 64
memory = 2 # for 1 / 2 wires used for memory
model = model.to(device)
for epoch in range(epochs):
    optimizer.zero_grad()
    x = torch.randint(0, 2, size=(batch_size,) + tuple(s // memory for s in model.shape), device=device, dtype=torch.float)
    pred = model(x)
    pred_circuit = model.apply(x)
    true = target(x)
    loss = bce(pred, true)
    loss.backward()
    optimizer.step()
    if not epoch % 100:
        inputs = x[0]
        circuit_prediction = pred_circuit[0]
        true_output = true[0]
        accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
        this_loss = loss.item()
        inputs = inputs.detach().cpu().numpy()
        circuit_prediction = circuit_prediction.detach().cpu().numpy()
        true_output = true_output.detach().cpu().numpy()
        print('inpt:', inputs)
        print('pred:', circuit_prediction)
        print('true:', true_output)
        print('%accuracy:', accuracy)
        print('loss:', loss.item())

