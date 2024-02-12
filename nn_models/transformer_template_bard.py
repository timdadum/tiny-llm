import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GPT, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerBlock(embedding_dim) for _ in range(num_layers)])
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        for layer in self.transformer_layers:
            embeds = layer(embeds)
        logits = self.linear(embeds)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, input):
        attn_output, attn_weights = self.attention(input, input, input)
        out = self.norm1(input + attn_output)
        out = self.ffn(out)
        out = self.norm2(out + input)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(MultiHeadAttention, self).__init__()

        self.d_k = hidden_dim // 8
        self.num_heads = 8

        self.q_linear = nn.Linear(hidden_dim, hidden_dim * self.num_heads)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim * self.num_heads)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim * self.num_heads)

        self.attention = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(hidden_dim * self.num_heads, hidden_dim)

    def forward(self, query, key, value):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q_splits = q.chunk(self.num_heads, dim=-1)
        k_splits = k.chunk(self.num_heads, dim=-1)
        v_splits = v.chunk(self.num_heads, dim=-1)

        attn = torch.matmul(q_splits, k_splits.transpose(-1, -2))
        attn = attn / math.sqrt(self.d_k)
        attn = self.attention(attn)

        out = torch.cat(torch.matmul(attn, v_splits) for attn, v_splits in zip(attn, v_splits), dim=-1)
        out = self.out_linear(out.reshape(out.size(0), -1))
        return out, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, input):
        out = self.fc1(input)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        return out

def main():
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 2
