import json
import utils.train as train
import torch.nn as nn
import torch
import nn_models.baseline_rnn as baseline
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE

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
pickle_name = 'shakespeare_short.pkl'
train_set, test_set, encoding, decoding = train.load_data(pickle_name)
train_loader, test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False), DataLoader(test_set, batch_size=batch_size, shuffle=False)
tokenizer = Tokenizer.from_file("tiny-llm/tokenizers/shakespeare_short.json")

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}')

# Train or load earlier state
model_name = 'shakespeare_short'
train.run(model, train_loader, test_loader, epochs, optim, loss_function, device, model_name)
# model.load_state_dict(torch.load('tiny-llm/trained_models/' + model_name))
model.eval()

# Test results
prompt = "King Arthur: Thou shalln't be named by the higher orders..."
result = model.sample(prompt, tokenizer, sequence_length=16, generation_length=500)

print(result)