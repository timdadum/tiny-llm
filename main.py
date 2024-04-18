import json
import utils.train as train
from tokenizers import Tokenizer
from tokenizers.models import BPE
import torch.nn as nn
import torch
import model_classes.gpt as gpt
from torch.utils.data import DataLoader
from utils.tokenization import load_tokenizer

# Load configuration file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Instantiate model
baseline_params = config['Hyperparameters']
train_params = config['Train_Params']

# Define model, tokenizer, attach
model = gpt.GPT(**baseline_params)
tokenizer = load_tokenizer(config)
model.set_tokenizer(tokenizer, config)

# Training parameters
optim = torch.optim.Adam(model.parameters(), train_params['lr'])
loss_function = nn.CrossEntropyLoss()

# Load data
train_set, test_set = train.load_training_data(config)
train_loader, test_loader = DataLoader(train_set, batch_size=train_params['batch_size'], shuffle=False), DataLoader(test_set, batch_size=train_params['batch_size'], shuffle=False)

# Set device
device = config['Train_Params']['device']
print(f'Training device is {device}')

# Train or load earlier state
model_name = 'jane_austen'
train.run(model, train_loader, test_loader, config, optim, loss_function, device, model_name)
print("Finished!")