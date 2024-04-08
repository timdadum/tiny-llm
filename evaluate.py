import torch
from model_classes import baseline_rnn

def load_model(model_class, name, params):
    model_path = f'tiny-llm/trained_models/{name}'
    
    model = model_class(**params)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

name = 'debug'
params = {
    "embedding_dim": 32,
    "vocab_size": 128,
    "hidden_size": 64,
    "num_layers": 2
}
trained_model = load_model(baseline_rnn.BaselineModel, name, params).to('cpu')

# Test results
prompt = "THE SONNETS 1 From fairest creatures we desire increase, That thereby beautys rose might never die, But as the riper should by time decease, His tender heir might bear his memory:"
trained_model.set_tokenizer('debug')
result = trained_model.sample(prompt, sequence_length=16, generation_length=64)

print(result)