import json
import utils.train as train
import torch.nn as nn
import torch
import model_classes.gpt as gpt
from torch.utils.data import DataLoader

# Load configuration file
with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

# Instantiate model
baseline_params = config['Hyperparameters']
train_params = config['Train_Params']

# Define tokenizer and model
model = gpt.GPT(**baseline_params)
tokenizer = gpt.GPTTokenizer()
tokenizer.from_file(config['Files']['tokenizer'])
model.set_tokenizer(tokenizer)

# Training parameters
optim = torch.optim.Adam(model.parameters(), train_params['lr'])
loss_function = nn.CrossEntropyLoss()

# Load data
train_set, test_set = train.load_data(config['Files']['data'])
train_loader, test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False), DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Training device is {device}')

# Train or load earlier state
model_name = 'brown'
train.run(model, train_loader, test_loader, epochs, optim, loss_function, device, model_name)
print("Finished!")