import torch.nn as nn
import torch

class BaselineModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.h = hidden_size
        self.n = num_layers

        self.lstm = nn.LSTM(embedding_dim, self.h, self.n, batch_first=True)

        self.linear = nn.Linear(self.h, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        
        # Initialize hidden and cell state with zeros: no information pertained at start of pass
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(embed, (h0, c0))
        logits = self.linear(out)[:, -1, :]

        return logits