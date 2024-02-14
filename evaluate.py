import torch
from model_classes import baseline_rnn

def load_model(model_class, name, params):
    model_path = f'tiny-llm/trained_models/{name}'
    
    model = model_class(**params)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

name = 'shakespeare_short'
params = {
    "embedding_dim": 32,
    "vocab_size": 256,
    "hidden_size": 64,
    "num_layers": 2
}
trained_model = load_model(baseline_rnn.BaselineModel, name, params).to('cpu')

# Test results
prompt = "King Arthur: Thou shalln't be named by the higher orders..."
trained_model.set_tokenizer('shakespeare_short')
result = trained_model.sample(prompt, sequence_length=16, generation_length=16)

print(result)