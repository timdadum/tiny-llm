import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import unittest
import torch.testing as torch_testing
import json
import re

# NO DROPOUT AND SKIP-CONNECTIONS YET!
    
class MultiHeadAttention(nn.Module):
    def __init__(self, k, heads):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads

        # Create extended mappings to key, query and value
        self.Wq = nn.Linear(k, k * heads, bias=False)
        self.Wk = nn.Linear(k, k * heads, bias=False)
        self.Wv = nn.Linear(k, k * heads, bias=False)

        # Unify heads back to k
        self.unify = nn.Linear(k * heads, k)

    def forward(self, x):
        # Infer quantities: assumes ([B] batch, [T] sequence length, [k] embedding dimension)
        b, t, k = x.size()
        h = self.heads

        # Project input to qkv, use add h dimension for head 
        queries = self.Wq(x).view(b, t, h, k)
        keys = self.Wk(x).view(b, t, h, k)
        values = self.Wv(x).view(b, t, h, k)

        # Combine batch and head dimension
        queries = queries.transpose(1,2).reshape(b*h, t, k)
        keys = keys.transpose(1,2).reshape(b*h, t, k)
        values = values.transpose(1,2).reshape(b*h, t, k)

        # Apply self-attention
        w = torch.bmm(queries, keys.transpose(1,2)) / k**0.5
        w_norm = F.softmax(w, dim=2)
        y = torch.bmm(w_norm, values).view(b, h, t, k)

        # Undo batch/head combination, unify heads for linear 
        y = y.transpose(1,2).reshape(b, t, k*h)
        y = self.unify(y)

        return y

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):

        super(TransformerBlock, self).__init__()

        self.att = MultiHeadAttention(k, heads=heads)
        self.dropout1 = nn.Dropout(p=0.2)
        self.norm1 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.GELU(),
            nn.Linear(4*k, k)
        )
        self.dropout2 = nn.Dropout(p=0.2)
        self.norm2 = nn.LayerNorm(k)

    def forward(self, x):
        # Apply self-attention and dropout
        att = self.att(x)
        att = self.dropout1(att)

        # Skip connection for improved flow
        att += x

        # Normalize
        att_norm = self.norm1(att)

        # Feed-forward and dropout
        ff = self.ff(att_norm)
        ff = self.dropout2(ff)

        # Skip connection again
        ff += att_norm

        # Normalize
        y = self.norm2(ff)

        return y

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding module

    k: embedding dimension
    t: sequence length
    """
    def __init__(self, k):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.k = k

    def forward(self, t, device):
        # Define sequence positions and common division term
        pos = torch.arange(0, t, device=device).unsqueeze(1)
        div = 10000**(2*torch.arange(0 ,self.k, 2, device=device)/self.k)
        
        # Fill sine and cosine encodings
        y = torch.zeros(t, self.k, device=device)
        y[:,0::2] = torch.sin(pos/div)
        y[:,1::2] = torch.cos(pos/div)

        return y

class GPTTokenizer:
    def __init__(self):
        self.encoding = None
        self.decoding = None
        self.vocab_size = None

    def from_file(self, path):
        """Load tokenizer encoding from a .json file."""
        with open(path, 'r') as file:
            self.encoding = json.load(file)
            self.decoding = {j: i for i, j in self.encoding.items()}
            self.vocab_size = len(self.encoding) + 1

    def tokenize(self, text, replace=True):
        text = text.lower()
        tokens = np.array(re.findall(r"\b\w+'\w+|\w+|[^\w\s]", text))

        if replace:
            # Replace unknown tokens
            tokens = np.array([token if token in self.encoding.keys() else '<UNK>' for token in tokens])
        return tokens

    def fit(self, corpus, unk_threshold=5, encode=False):
        """Determine the encoding from the corpus."""
        tokens = self.tokenize(corpus, replace=False)

        unique, counts = np.unique(tokens, return_counts=True)
        rare_tokens = unique[counts < unk_threshold]
        
        # Mask rare words as '<UNK>', redefine unique tokens
        tokens = np.where(np.isin(tokens, rare_tokens), '<UNK>', tokens)

        unique = np.unique(tokens)
        
        # Create mapping with '<UNK>' as a special case
        mapping = {'<UNK>': 0}  # Start with '<UNK>' token
        for i, token in enumerate(unique, start=1):
          mapping[token] = i

        print(list(mapping.items())[-10:])  # Print the last 10 mappings to check indices

        # Round up vocabulary
        vocab_size = len(mapping) + 1
        self.encoding = mapping
        self.decoding = {j: i for i, j in mapping.items()}
        self.vocab_size = vocab_size
        print(f"Tokenizer fit with vocab size {len(self.encoding)}")

        # Encode after fitting if required
        if encode:
            return self.encode(tokens)

    def encode(self, tokens):
        """Encode a text using the tokenizer's encoding."""
        if self.encoding is None:
            raise ValueError("Encoding has not been set. Please load an encoding or fit the tokenizer.")
  
        encoded_text = [self.encoding[token] for token in tokens]
        encoded_tensor = torch.tensor(encoded_text, dtype=torch.long)
        return encoded_tensor

    def decode(self, tokens):
        decoded_tokens = [self.decoding[int(token)] for token in tokens]
        decoded_text = ' '.join(decoded_tokens)
        return decoded_text
    
    def save(self, path):
        """Saves tokenizer (or rather - its mapping which characterizes the tokenizer) to provided path"""
        with open(path, 'w') as file:
            json.dump(self.encoding, file)
        f"""Tokenizer succesfully saved at {path}"""

class GPT(nn.Module):
    def __init__(self, k=128, heads=2, blocks=2, device=None):
        super(GPT, self).__init__()
        # Explicitly define device during declaration
        self.device = device if device else torch.device('cpu')
        
        # Instantiate (or skeleton) components
        self.embed = None
        self.pos_encoding = SinusoidalPositionalEncoding(k).to(device)
        self.dropout1 = nn.Dropout(p=0.2)
        self.transformers = nn.ModuleList([TransformerBlock(k, heads).to(self.device) for i in range(blocks)])
        self.norm = nn.LayerNorm(k).to(device)
        self.unembed = None

        # Save hyperparameters
        self.k = k
        self.heads = heads
        self.blocks = blocks

        # Create inference-time tokenizer
        self.tokenizer = None

    def check_device_of_components(self):
        # Using named_modules to capture all modules in the hierarchy
        for name, module in self.named_modules():
            # Attempt to get a parameter to infer device. If no parameters, skip.
            try:
                param = next(module.parameters())
                print(f"{name} is on {param.device}")
            except StopIteration:
                # This module has no parameters
                if name:  # Skip printing for the top-level module which has name ''
                    print(f"{name} has no parameters")

        # Check for tensors directly attached to this module (e.g., custom buffers not registered as buffers)
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                print(f"Tensor {name} is on {attr.device}")
    
    def forward(self, x):
        if self.embed is None or self.unembed is None:
            raise ValueError('Embeddings are not set. Did you set a tokenizer yet?')

        # Embed input
        x = self.embed(x.long())
        _, t, _ = x.size()

        # Add positional encoding
        x += self.pos_encoding(t, device=self.device)
        
        # Apply series of transformer blocks
        for transformer in self.transformers:
            x = transformer(x)

        # Normalize
        x = self.norm(x).squeeze(0)

        # Unembed output
        y = self.unembed(x)
        return y
    
    def set_tokenizer(self, tokenizer):
        """Takes a GPTTokenizer class and assigns to class for convenience"""
        if not isinstance(tokenizer, GPTTokenizer):
            raise ValueError("Tokenizer is not of class GPTTokenizer")
        self.tokenizer = tokenizer

        # Set layers accordingly, place on device
        self.embed = nn.Embedding(self.tokenizer.vocab_size, self.k).to(self.device)
        self.unembed = nn.Linear(self.k, self.tokenizer.vocab_size).to(self.device)

        print("Tokenizer succesfully set")

    def sample(self, x, T=1.0, generation_length=128):
        """Generates new text based on input prompt"""
        # TODO: Perhaps handle batch sampling too?
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided. Please provide a GPTTokenizer")

        # Encode input, place on device
        tokens = self.tokenizer.tokenize(x)
        encodings = self.tokenizer.encode(tokens)
        encodings = encodings.to(self.device)

        # Batchify
        encodings = encodings.unsqueeze(0)

        # incrementally add tokens
        for i in range(generation_length):
            # Forward pass
            y = self(encodings)

            # Index last-in-sequence probabilities, apply temperature and softmax to output probs
            y = y[-1,:] / T
            probs = torch.softmax(y, dim=0)

            # Predict from resulting probability distribution
            pred = torch.multinomial(probs, num_samples=1)

            # Append predicted token to encodings
            encodings = torch.cat((encodings, pred.unsqueeze(0)), dim=1)
        
        # Unbatchify and decode
        encodings = encodings[0]
        result = self.tokenizer.decode(encodings)

        return result