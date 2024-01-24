import json
import utils.preprocess as pre
import utils.train as train
import torch.nn as nn
import torch
import models.baseline_rnn

# Load the JSON file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Load data
pickle_name = 'shakespeare_short.pkl'
train_loader, test_loader = train.load_data(pickle_name)

# Instantiate model
model = Bas
optimizer = nn.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss() 

# Train
