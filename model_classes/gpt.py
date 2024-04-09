import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import unittest
import torch.testing as torch_testing

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
        self.ff = nn.Sequential([
            nn.Linear(k, 4*k),
            nn.GELU(),
            nn.Linear(4*k, k)
        ])
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

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding module

    k: embedding dimension
    t: sequence length
    """
    def __init__(self, k):
        super(SinusoidalPositionalEncoding).__init__()
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


class GPT(nn.Module):
    def __init__(self, k=128, heads=2, blocks=2, vocabulary_size=1028):
        super(GPT, self).__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(k)
        self.transformers = nn.Sequential([
            TransformerBlock(k, heads) for i in range(blocks)
        ])
        self.norm = nn.LayerNorm(k)
        self.ff = nn.Linear(k, vocabulary_size)

    def forward(self, x):
        _, t, _ = x.size()

        # Add positional encoding
        x += self.pos_encoding(t)

        # Apply series of transformer blocks
        attended = self.transformers(x)

        # Normalize
        attended_normalized = self.norm(attended)

        # Feed forward and return
        y = self.ff(attended_normalized)
        
        return y
    
    def sample(self, y, T=1.0):
        """Samples a character from the forward output of this class"""
        pass

### TESTS
class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        batch_size, seq_length, embedding_dim, heads = 2, 10, 64, 4
        mha = MultiHeadAttention(embedding_dim, heads)
        x = torch.rand(batch_size, seq_length, embedding_dim)  # Dummy input tensor
        output = mha(x)
        expected_shape = (batch_size, seq_length, embedding_dim)
        self.assertEqual(output.shape, expected_shape)

class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_output_shape(self):
        embedding_dim = 64
        seq_length = 10
        spe = SinusoidalPositionalEncoding(embedding_dim)
        output = spe.forward(seq_length)
        expected_shape = (seq_length, embedding_dim)
        self.assertEqual(output.shape, expected_shape)

class TestSinusoidalPositionalEncodingValues(unittest.TestCase):
    def test_first_position(self):
        embedding_dim = 64
        spe = SinusoidalPositionalEncoding(embedding_dim)
        output = spe.forward(1)  # Only get the first position
        # You can check specific properties, like whether the first value is close to expected
        # This is a simplistic check; you might have more complex checks based on your expectations
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([0.0]), atol=1e-6))

if __name__ == '__main__':
    unittest.main()