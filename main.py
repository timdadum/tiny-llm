import json
import utils.train as train
import torch.nn as nn
import torch
import model_classes.baseline_rnn as baseline
from torch.utils.data import DataLoader

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Instantiate model
baseline_params = config['Baseline_Hyperparams']
train_params = config['Train_Params']

lr = train_params['lr']
epochs = train_params['epochs']
batch_size = train_params['batch_size']

model = baseline.BaselineModel(**baseline_params)
optim = torch.optim.Adam(model.parameters(), train_params['lr'])
loss_function = nn.CrossEntropyLoss()

# Load data
pickle_name = 'debug.pkl'
train_set, test_set = train.load_data(pickle_name)
train_loader, test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False), DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}')

# Train or load earlier state
model_name = 'debug'
train.run(model, train_loader, test_loader, epochs, optim, loss_function, device, model_name)
print("Finished!")