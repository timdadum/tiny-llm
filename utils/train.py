import pickle as pk
import torch.nn as nn
import torch
import os

def load_data(pickle_name):
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "..", 'data', pickle_name)

    if not os.path.isfile(data_path):
        raise ValueError("[train.load_data]: Path not found")

    with open(data_path, 'rb') as dat:
        train_loader, test_loader = pk.load(dat)
    return train_loader, test_loader

def train(model, loader, epochs, optim, loss_function, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optim.zero_grad()
            
            X, y = batch
            X, y = X.to(device), y.to(device)
            out = model(X)

            loss = loss_function(out, y)
            loss.backward()
            total_loss += loss.item()
            
            optim.step()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epochs + 1}/{epochs} loss: {avg_loss:.4f}")
        

def test(model, test_loader, loss_function, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = loss_function(outputs, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

def run(model, train_loader, test_loader, epochs, lr, optim, loss_function, device):

    model.to(device)

    train(model, train_loader, epochs, optim, loss_function, device)
    test(model, test_loader, loss_function, device)

    print("Finished!")