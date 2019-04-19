import itertools
import numpy
import torch

checkpoint = 100

def stochastic(model, target, shape, epochs):
    bce = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    rolling_loss = 0
    rolling_accuracy = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = torch.randint(0, 2, size=shape, device=model.device, dtype=torch.float)
        pred = model(x)
        pred_circuit = model.apply(x)
        true = target(x)
        loss = bce(pred, true)
        rolling_loss *= (1 - 1. / checkpoint)
        rolling_loss +=  1. / checkpoint * loss.item()
        accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
        rolling_accuracy *= (1 - 1. / checkpoint)
        rolling_accuracy +=  1. / checkpoint * accuracy
        loss.backward()
        optimizer.step()
        if not epoch % checkpoint:
            inputs = x[0]
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

def batch(model, target, shape, epochs):
    bce = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = numpy.asarray(list(itertools.product(range(2), repeat=shape[1])))
        numpy.random.shuffle(x)
        x = torch.from_numpy(x).to(model.device).float()
        pred = model(x)
        pred_circuit = model.apply(x)
        true = target(x)
        loss = bce(pred, true)
        accuracy = (1 - abs(true - pred_circuit).max(1)[0]).mean().item() * 100
        loss.backward()
        optimizer.step()
        if not epoch % checkpoint:
            inputs = x[0]
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
            print('%accuracy:', accuracy)
            print('loss:', loss.item())
