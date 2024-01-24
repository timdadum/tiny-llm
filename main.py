import json
import utils.train as train
import torch.nn as nn
import torch
import models.baseline_rnn as baseline

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Load data
pickle_name = 'debug.pkl'
train_loader, test_loader, encoding, decoding, tokenizer = train.load_data(pickle_name)

# Instantiate model
baseline_params = config['Baseline_Hyperparams']
train_params = config['Train_Params']

lr = train_params['lr']
epochs = train_params['epochs']

model = baseline.BaselineModel(**baseline_params)
optim = torch.optim.Adam(model.parameters(), train_params['lr'])
loss_function = nn.CrossEntropyLoss()

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Train
train.run(model, train_loader, test_loader, epochs, optim, loss_function, device)

# Test results
prompt = "King Arthur: Thou shall'nt be named by the higher orders..."
model.sample(prompt)
