import torch
from model_classes import gpt
from utils.tokenization import load_tokenizer
import json

with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

def load_model(model_class, config):
    model = model_class(**config['Hyperparameters'])
    tokenizer = load_tokenizer(config)
    model.set_tokenizer(tokenizer, config)
    model.load_state_dict(torch.load(config['Files']['model'], map_location=config['Train_Params']['device']))
    model.eval()
    return model

trained_model = load_model(gpt.GPT, config)

# Test results
prompt = "badger badger"
result = trained_model.sample(prompt)
print(f'result is {result}')