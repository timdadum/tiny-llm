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
prompt = "The lightning struck"

# DEBUG
# Example text
test_sentence = "The lightning struck."
encoded = trained_model.tokenizer.encode(test_sentence)
print("Encoded Tokens:", encoded.tokens)
print("Encoded IDs:", encoded.ids)

# Decode the output
decoded_output = trained_model.tokenizer.decode(encoded.ids)
print("Decoded Output:", decoded_output)

result = trained_model.sample(prompt)
print(f'result is {result}')