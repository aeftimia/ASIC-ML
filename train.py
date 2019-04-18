import torch

def train(model, target, shape, epochs):
    bce = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = torch.randint(0, 2, size=shape, device=model.device, dtype=torch.float)
        pred = model(x)
        pred_circuit = model.apply(x)
        true = target(x)
        loss = bce(pred, true)
        accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
        loss.backward()
        if accuracy < 100:
            optimizer.step()
        if not epoch % 100:
            inputs = x[0]
            circuit_prediction = pred_circuit[0]
            true_output = true[0]
            this_loss = loss.item()
            inputs = inputs.detach().cpu().numpy()
            circuit_prediction = circuit_prediction.detach().cpu().numpy()
            true_output = true_output.detach().cpu().numpy()
            print('inpt:', inputs)
            print('pred:', circuit_prediction)
            print('true:', true_output)
            print('%accuracy:', accuracy)
            print('loss:', loss.item())
