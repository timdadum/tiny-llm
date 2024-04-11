import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import unittest
import torch.testing as torch_testing
import json

# NO DROPOUT AND SKIP-CONNECTIONS YET!

"""
class SelfAttention(nn.Module):
    def __init__(self, k):
        super(SelfAttention, self).__init__()

        self.k = k

        # Self-attention
        self.Wq = nn.Linear(k, k, bias=False)
        self.Wk = nn.Linear(k, k, bias=False)
        self.Wv = nn.Linear(k, k, bias=False)

    def forward(self, x):
        # Assumes x ~ (batch size, sequence length, embedding dimension) 
        b, t, k = x.size()
        queries, keys, values = self.Wq(x), self.Wk(x), self.Wv(x)
        w = torch.bmm(queries, keys.transpose(1,2)) / np.sqrt(k)
        w_norm = F.softmax(w, dim=2)
        y = torch.bmm(w_norm, values)
        return y
"""
    
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
        self.norm1 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.GELU(),
            nn.Linear(4*k, k)
        )
        self.norm2 = nn.LayerNorm(k)

    def forward(self, x):
        # Apply self-attention
        att = self.att(x)

        # Normalize
        att_norm = self.norm1(att)

        # Feed-forward
        ff = self.ff(att_norm)

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

    def forward(self, t):
        # Define sequence positions and common division term
        pos = torch.arange(0, t).unsqueeze(1)
        div = 10000**(2*torch.arange(0 ,self.k, 2)/self.k)
        
        # Fill sine and cosine encodings
        y = torch.zeros(t, self.k)
        y[:,0::2] = torch.sin(pos/div)
        y[:,1::2] = torch.cos(pos/div)

        return y

class GPTTokenizer:
    def __init__(self):
        self.mapping = None
        self.vocab_size = None

    def from_file(self, path):
        """Load tokenizer encoding from a .json file."""
        with open(path, 'r') as file:
            self.mapping = json.load(file)
            self.vocab_size = len(self.mapping)
        print("Tokenizer encoding loaded from file!")

    def fit(self, corpus, unk_threshold=1e-4, encode=False):
        """Determine the encoding from the corpus."""
        tokens = np.array(corpus.split())
        unique, counts = np.unique(tokens, return_counts=True)
        threshold = np.round(unk_threshold * len(tokens))
        rare_tokens = unique[counts < threshold]
        
        # Mask rare words as '<UNK>'
        tokens = np.where(np.isin(tokens, rare_tokens), '<UNK>', tokens)
        unique = np.unique(tokens)
        self.vocab_size = len(unique)
        
        # Create mapping
        self.mapping = {token: i for i, token in enumerate(unique)}
        print(f"Tokenizer fit with vocab size {len(self.mapping)}")

        if encode:
            tokens = self.encode(corpus)
            return tokens

    def encode(self, text):
        """Encode a text using the tokenizer's mapping."""
        if self.mapping is None:
            raise ValueError("Encoding has not been set. Please load an encoding or fit the tokenizer.")
        
        tokens = np.array(text.split())
        vectorized_map = np.vectorize(lambda x: self.mapping.get(x, self.mapping.get('<UNK>', -1)))
        encoded_text = vectorized_map(tokens)
        encoded_tensor = torch.tensor(encoded_text, dtype=torch.long)
        return encoded_tensor

    def decode(self, tokens):
        # Assuming self.mapping is {token_id: token_str}
        decoded_tokens = [self.mapping.get(token, '<UNK>') for token in tokens]
        decoded_text = ' '.join(decoded_tokens)
        return decoded_text
    
    def save(self, path):
        """Saves tokenizer (or rather - its mapping which characterizes the tokenizer) to provided path"""
        with open(path, 'w') as file:
            json.dump(self.mapping, file)
        f"""Tokenizer succesfully saved at {path}"""

class GPT(nn.Module):
    def __init__(self, k=128, heads=2, blocks=2, device=None):
        super(GPT, self).__init__()
        # Explicitly define device during declaration
        self.device = device if device else torch.device('cpu')
        
        # Instantiate (or skeleton) components
        self.embed = None
        self.pos_encoding = SinusoidalPositionalEncoding(k).to(device)
        self.transformers = [TransformerBlock(k, heads).to(device) for i in range(blocks)]
        self.norm = nn.LayerNorm(k)
        self.unembed = None

        # Save hyperparameters
        self.k = k
        self.heads = heads
        self.blocks = blocks

        # Create inference-time tokenizer
        self.tokenizer = None
    
    def forward(self, x):
        if self.embed is None or self.unembed is None:
            raise ValueError('Embeddings are not set. Did you set a tokenizer yet?')

        x = self.embed(x.long())
        _, t, _ = x.size()

        # Add positional encoding
        x += self.pos_encoding(t)
        
        # Apply series of transformer blocks
        for transformer in self.transformers:
            x = transformer(x)

        # Normalize
        x = self.norm(x)

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
        
        # encode input
        tokens = self.tokenizer.encode(x)

        # batchify
        tokens = tokens.unsqueeze(0)

        # incrementally add tokens
        for i in range(generation_length):
            # Forward pass
            y = self(tokens)

            # Index last-in-sequence probabilities, apply temperature and softmax to output probs
            y = y[0,-1,:] / T
            probs = torch.softmax(y, dim=0)

            # Predict from resulting probability distribution
            pred = torch.multinomial(probs, num_samples=1)

            # Append predicted token to tokens
            tokens = torch.cat((tokens, pred.unsqueeze(0)), dim=1)
        
        # unbatchify and decode
        tokens = tokens[0]
        result = self.tokenizer.decode(tokens)

        return result