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
    model.check_device_of_components()
    model.load_state_dict(torch.load(config['Files']['model'], map_location=config['Hyperparameters']['device']))
    model.eval()
    return model

trained_model = load_model(gpt.GPT, config)

# Test results
prompt = "In the State of Texas, President Kennedy addressed the public on the importance of healthcare reform. He emphasized that every citizen deserves access to medical care, regardless of their income. The administration's new plan aims to increase funding for local hospitals and introduce a program to lower the cost of prescription drugs. As debates in the House and Senate continue, Mr. Kennedy urged lawmakers to prioritize the well-being of the American people over partisan politics"
result = trained_model.sample(prompt)

print(f'result is {result}')