import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.preprocess import bpe_preprocess, bpe_postprocess
from torch.cuda.amp import autocast
from tokenizers import Tokenizer

class BaselineModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = hidden_size
        self.n = num_layers

        self.lstm = nn.LSTM(embedding_dim, self.h, self.n, batch_first=True, dropout=0.5)

        self.linear = nn.Linear(self.h, vocab_size)

        self.tokenizer = None 

    def forward(self, x):
        # Check if the model is on a CUDA device
        if next(self.parameters()).is_cuda:
            # Use autocast only on CUDA devices for mixed precision
            with autocast():
                return self._forward_impl(x)
        else:
            # No autocast for CPU or when CUDA is not available
            return self._forward_impl(x)

    def _forward_impl(self, x):
        # Actual forward function
        device = x.device
        x = torch.Tensor.long(x)
        embed = self.embedding(x)
        
        # Initialize hidden and cell state with zeros
        h0 = torch.zeros(self.n, x.size(0), self.h).to(device)
        c0 = torch.zeros(self.n, x.size(0), self.h).to(device)

        out, _ = self.lstm(embed, (h0, c0))
        logits = self.linear(out)[:, -1, :]
        return logits
    
    def set_tokenizer(self, name):
        self.tokenizer = Tokenizer.from_file(f'tiny-llm/tokenizers/{name}.json')
        print("Tokenizer succesfully set")

    def sample(self, prompt: str, sequence_length, generation_length, temperature: int=1.0):
        if not self.tokenizer:
            raise ValueError("Please set tokenizer first using .set_tokenizer()")
        
        # Encode input
        preprocessed_prompt = bpe_preprocess(prompt, save=False)
        encodings = self.tokenizer.encode(preprocessed_prompt)
        input_ids = encodings.ids
        
        # Debug: print subwords to verify
        print(f"Prompt in subwords: {encodings.tokens}")
        
        # Pad prompt if provided input sequence is too short
        if len(input_ids) > sequence_length:
            input_ids = input_ids[-sequence_length:]
        elif len(input_ids) < sequence_length:
            padding = torch.zeros((1, sequence_length - len(input_ids)), dtype=torch.long)
            input_ids = torch.cat((padding, input_ids), dim=1)

        # Initialie the tensor for generated IDs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        generated_ids = torch.tensor(input_ids, dtype=torch.float32).to(device)

        # Iteratively generate new subword
        for _ in range(generation_length):
            logits = self(generated_ids.unsqueeze(0)).squeeze(0) # First get output logits

            # Apply temperature scaling, sample from prediction
            softmaxed_logits = F.softmax(logits / temperature, dim=-1)
            sampled_token_id = torch.multinomial(softmaxed_logits, num_samples=1)
            generated_ids = torch.cat((generated_ids, sampled_token_id))

        # Decode integer encodings and clean decoded text
        raw_text = self.tokenizer.decode(generated_ids.long().squeeze().tolist())
        clean_text = bpe_postprocess(raw_text)
        return clean_text