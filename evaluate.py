import torch
from model_classes import gpt
from tokenizers.models import BPE
from tokenizers import Tokenizer
import json

with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

def load_model(model_class, config):
    model = model_class(**config['Hyperparameters'])
    tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
    tokenizer = tokenizer.from_file(config['Files']['tokenizer'])
    
    model.set_tokenizer(tokenizer, config)
    model.load_state_dict(torch.load(config['Files']['model'], map_location=config['Hyperparameters']['device']))
    model.eval()
    return model

trained_model = load_model(gpt.GPT, config)

# Test results
prompt = "lightning _illuminates _the _sky, _followed _by _the _deep _rumble _of _thunder _rolling _across _the"
result = trained_model.sample(prompt)

print(f'result is {result}')