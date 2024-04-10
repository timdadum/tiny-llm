import torch
from model_classes import gpt
import json

with open('tiny-llm/PARAMS.json', 'r') as file:
    config = json.load(file)

def load_model(model_class, config):
    model = model_class(**config['Hyperparameters'])
    tokenizer = gpt.GPTTokenizer()
    tokenizer.from_file(config['Files']['tokenizer'])
    model.set_tokenizer(tokenizer)
    model.load_state_dict(torch.load(config['Files']['model'], map_location='cpu'))
    model.eval()
    return model

trained_model = load_model(gpt.GPT, config).to('cpu')

# Test results
prompt = "The American soliders that have landed in the far west have seldom been more isolated. Their opponents try to seize the eastern land but have not yet succeeded in doing so...."
result = trained_model.sample(prompt)

print(f'result is {result}')