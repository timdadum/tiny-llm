import pickle as pk
import torch
import torch.nn as nn
import os
from tokenizers import Tokenizer

def get_file_path(relative_path):
    """Generates absolute file path from a relative path."""
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, relative_path)

def load_data(pickle_name):
    """Loads training and testing data from a pickle file."""
    data_path = get_file_path(f'../data/{pickle_name}')

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f'File not found at {data_path}')

    with open(data_path, 'rb') as dat:
        train_set, test_set, encoding, decoding = pk.load(dat)
        
    return train_set, test_set, encoding, decoding

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def train_one_epoch(model, train_loader, optim, loss_function, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        optim.zero_grad()
        X, y = to_device(batch, device)
        out = model(X)

        loss = loss_function(out, y.long())
        loss.backward()
        total_loss += loss.item()
        optim.step()
    
    return total_loss / len(train_loader)

def train(model, train_loader, test_loader, epochs, optim, loss_function, device):
    """Trains and tests the model for a given number of epochs."""
    counter = 0
    lowest_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optim, loss_function, device)
        print(f'[TR] Epoch {epoch + 1}/{epochs} loss: {train_loss:.4f}')

        test_loss = test(model, test_loader, loss_function, device)
        print(f'[TE] Test Loss: {test_loss:.4f}')

        # Handle early stopping
        if test_loss < lowest_loss:
            counter = 0
            lowest_loss = test_loss
        else:
            counter += 1

        if counter > 10:
            print(f"Early stopping after {epoch} epochs... Final loss is {lowest_loss}")
        break
    

def test(model, test_loader, loss_function, device):
    """Tests the model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            X, y = to_device(batch, device)
            out = model(X)
            loss = loss_function(out, y.long())
            total_loss += loss.item()

    return total_loss / len(test_loader)

def run(model, train_loader, test_loader, epochs, optim, loss_function, device, model_name=None):
    """Runs the training and testing process, and saves the model if requested."""
    model.to(device)
    train(model, train_loader, test_loader, epochs, optim, loss_function, device)

    print('Finished!')

    if model_name:
        save_model(model, model_name)

def save_model(model, model_name):
    """Saves the model to a file."""
    model_path = get_file_path(f'../trained_models/{model_name}')
    torch.save(model.state_dict(), model_path)
    print(f'Saved model at {model_path}')